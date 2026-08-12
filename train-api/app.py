# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : microservice FastAPI (port 5010) qui declenche et
# supervise l'entrainement (regression DJU + desaisonnalisation
# + Holt-Winters + SARIMA), calcule le backtest 2019, choisit le
# meilleur modele et l'expose via /metrics (Prometheus) et MLflow.
# Pattern d'architecture (job registry async, Gauges Prometheus,
# quality-gate en subprocess pytest) repris de
# rakuten_mlops_services/train-api/app.py, adapte a un pipeline
# scikit/statsmodels mono-processus (pas besoin d'isoler un
# subprocess TensorFlow ici).
#
# Fonctions principales :
#   - _run_training_pipeline(job_id) : orchestre data_loader ->
#     preprocess -> split train/test(2019) -> fit HW + SARIMA ->
#     evaluate -> selection du meilleur -> refit sur toute la
#     serie -> artifacts.save_artifacts() -> log MLflow -> met a
#     jour les Gauges Prometheus
#   - POST /train (202) : cree un job_id, refuse (409) si un job
#     tourne deja, lance _run_training_pipeline en arriere-plan
#   - GET /train/status/{job_id} : etat/resultat d'un job
#   - POST /quality-gate : execute pytest tests/test_model_quality.py
#   - GET /health, GET /metrics
#
# Variables / constantes importantes :
#   - PARAMS_PATH="params.yaml", _training_jobs (registre en memoire)
#   - Gauges Prometheus : model_mape, model_rmse (labels model="holt_winters"|"sarima"),
#     model_best_selected (1 pour le modele retenu)
#
# Dependances externes : fastapi, pydantic, prometheus_client,
# prometheus_fastapi_instrumentator, pyyaml, mlflow (optionnel),
# services.* (interne), utils.logger
# ============================================================
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from services import artifacts, data_loader, evaluate, preprocess, trainer_holtwinters, trainer_sarima
from utils.logger import get_logger

logger = get_logger()
app = FastAPI(title="Train API — elec_prediction", description="Temperature correction + Holt-Winters/SARIMA training", version="1.0")

# --- Prometheus metrics ---
model_mape_gauge = Gauge("model_mape", "Backtest MAPE (%) on the held-out test year", ["model"])
model_rmse_gauge = Gauge("model_rmse", "Backtest RMSE on the held-out test year", ["model"])
model_best_gauge = Gauge("model_best_selected", "1 if this model was selected for deployment, else 0", ["model"])
ols_r2_gauge = Gauge("ols_temperature_r2", "R2 of the Consommation ~ DJU OLS regression")
last_train_timestamp = Gauge("train_last_run_timestamp", "Unix timestamp of the last completed training run")

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

PARAMS_PATH = os.getenv("PARAMS_PATH", "params.yaml")
JOB_DIR = os.getenv("JOB_DIR", "data/jobs")

_training_jobs: Dict[str, Dict[str, Any]] = {}


def _load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _load_persisted_jobs():
    """Recover job results from disk on startup — mirrors rakuten's crash-recovery pattern."""
    if not os.path.exists(JOB_DIR):
        return
    for fname in os.listdir(JOB_DIR):
        if not fname.endswith(".json"):
            continue
        job_id = fname[:-5]
        try:
            with open(os.path.join(JOB_DIR, fname)) as f:
                job = json.load(f)
            if job.get("status") == "running":
                job["status"] = "interrupted"
                job["error"] = "train-api restarted while job was running — re-trigger /train"
            _training_jobs[job_id] = job
        except Exception as e:
            logger.warning(f"Could not load persisted job {job_id}: {e}")


_load_persisted_jobs()


def _maybe_log_mlflow(params: dict, stats: dict, metrics: dict, best_model: str):
    """Best-effort MLflow logging — never fails the training run if MLflow/DagsHub is unreachable."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        logger.info("MLFLOW_TRACKING_URI not set — skipping MLflow logging")
        return
    try:
        import mlflow

        mlflow.set_experiment(params.get("mlflow", {}).get("experiment_name", "elec_prediction"))
        with mlflow.start_run():
            mlflow.log_params({
                "sarima_order": params["model_sarima"]["order"],
                "sarima_seasonal_order": params["model_sarima"]["seasonal_order"],
                "sarima_log_transform": params["model_sarima"]["log_transform"],
                "hw_trend": params["model_holt_winters"]["trend"],
                "hw_seasonal": params["model_holt_winters"]["seasonal"],
                "test_year": params["data"]["test_year"],
            })
            mlflow.log_metrics({
                "ols_r2": stats["ols_r2"],
                "holt_winters_mape": metrics["holt_winters"]["mape"],
                "holt_winters_rmse": metrics["holt_winters"]["rmse"],
                "sarima_mape": metrics["sarima"]["mape"],
                "sarima_rmse": metrics["sarima"]["rmse"],
            })
            mlflow.set_tag("best_model", best_model)
        logger.info("MLflow run logged successfully")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


def _run_training_pipeline(job_id: str):
    os.makedirs(JOB_DIR, exist_ok=True)
    job_file = os.path.join(JOB_DIR, f"{job_id}.json")

    try:
        params = _load_params()
        seasonal_periods = params["preprocess"]["seasonal_periods"]
        test_year = params["data"]["test_year"]

        # --- 1. Load + preprocess (OLS temperature correction + deseasonalization) ---
        df_merged = data_loader.load_merged(params["data"]["raw_conso_path"], params["data"]["raw_dju_path"])
        df, ols_results, stats = preprocess.run_preprocessing(df_merged, seasonal_periods=seasonal_periods)
        os.makedirs(os.path.dirname(params["data"]["processed_path"]), exist_ok=True)
        df.to_csv(params["data"]["processed_path"])
        ols_r2_gauge.set(stats["ols_r2"])

        # --- 2. Train/test split for backtesting (matches notebook's 2019 holdout) ---
        train_df = df[df.index.year < test_year]
        test_df = df[df.index.year == test_year]
        if len(test_df) == 0:
            raise ValueError(f"No data found for test_year={test_year} — cannot backtest")
        horizon = len(test_df)

        # --- 3. Holt-Winters backtest ---
        hw_cfg = params["model_holt_winters"]
        hw_model_bt = trainer_holtwinters.fit_holt_winters(
            train_df["Conso_correction"], seasonal_periods=seasonal_periods,
            trend=hw_cfg["trend"], seasonal=hw_cfg["seasonal"],
        )
        hw_pred = trainer_holtwinters.forecast(hw_model_bt, horizon)
        hw_metrics = evaluate.evaluate_forecast(test_df["Conso_correction"].values, hw_pred)

        # --- 4. SARIMA backtest (log-transformed, matches notebook's a-posteriori model) ---
        sarima_cfg = params["model_sarima"]
        sarima_results_bt = trainer_sarima.fit_sarima(
            train_df["Conso_correction"], order=sarima_cfg["order"],
            seasonal_order=sarima_cfg["seasonal_order"], log_transform=sarima_cfg["log_transform"],
        )
        sarima_pred, sarima_lo, sarima_hi = trainer_sarima.forecast(
            sarima_results_bt, horizon, log_transform=sarima_cfg["log_transform"],
        )
        sarima_metrics = evaluate.evaluate_forecast(test_df["Conso_correction"].values, sarima_pred)

        metrics = {"holt_winters": hw_metrics, "sarima": sarima_metrics}

        # --- 5. Select best model by MAPE ---
        best_model = "sarima" if sarima_metrics["mape"] <= hw_metrics["mape"] else "holt_winters"

        model_mape_gauge.labels(model="holt_winters").set(hw_metrics["mape"])
        model_rmse_gauge.labels(model="holt_winters").set(hw_metrics["rmse"])
        model_mape_gauge.labels(model="sarima").set(sarima_metrics["mape"])
        model_rmse_gauge.labels(model="sarima").set(sarima_metrics["rmse"])
        model_best_gauge.labels(model="holt_winters").set(1 if best_model == "holt_winters" else 0)
        model_best_gauge.labels(model="sarima").set(1 if best_model == "sarima" else 0)

        # --- 6. Refit both models on the FULL series for deployment (predict-api serves this) ---
        hw_model_full = trainer_holtwinters.fit_holt_winters(
            df["Conso_correction"], seasonal_periods=seasonal_periods,
            trend=hw_cfg["trend"], seasonal=hw_cfg["seasonal"],
        )
        sarima_results_full = trainer_sarima.fit_sarima(
            df["Conso_correction"], order=sarima_cfg["order"],
            seasonal_order=sarima_cfg["seasonal_order"], log_transform=sarima_cfg["log_transform"],
        )

        metadata = {
            "best_model": best_model,
            "sarima_log_transform": sarima_cfg["log_transform"],
            "seasonal_periods": seasonal_periods,
            "test_year": test_year,
            "metrics": metrics,
            "ols_stats": stats,
            "last_observed_month": str(df.index.max().date()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        history = {
            "job_id": job_id,
            "metrics": metrics,
            "best_model": best_model,
            "quality_gate": {
                "mape_threshold": params["quality_gate"]["mape_threshold"],
                "selected_mape": metrics[best_model]["mape"],
                "passes": metrics[best_model]["mape"] < params["quality_gate"]["mape_threshold"],
            },
        }

        artifacts.save_artifacts(ols_results, hw_model_full, sarima_results_full, metadata, history)
        _maybe_log_mlflow(params, stats, metrics, best_model)
        last_train_timestamp.set(datetime.now(timezone.utc).timestamp())

        result = {
            "status": "success",
            "job_id": job_id,
            "best_model": best_model,
            "metrics": metrics,
            "ols_stats": stats,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _training_jobs[job_id].update(result)
        with open(job_file, "w") as f:
            json.dump(_training_jobs[job_id], f, indent=2, default=str)

        logger.info(f"[job={job_id}] Training complete — best_model={best_model} mape={metrics[best_model]['mape']}")

    except Exception as exc:
        logger.exception(f"[job={job_id}] Training failed: {exc}")
        _training_jobs[job_id].update({
            "status": "failed",
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        with open(job_file, "w") as f:
            json.dump(_training_jobs[job_id], f, indent=2, default=str)


class TrainRequest(BaseModel):
    pass  # all configuration comes from params.yaml — kept as a body for future extension (e.g. override test_year)


@app.post("/train", status_code=202)
def train_model(_req: TrainRequest = TrainRequest()):
    running = [j for j in _training_jobs.values() if j.get("status") == "running"]
    if running:
        raise HTTPException(status_code=409, detail=f"Training job {running[0]['job_id']} already in progress")

    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {
        "status": "running",
        "job_id": job_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    thread = threading.Thread(target=_run_training_pipeline, args=(job_id,), daemon=True)
    thread.start()

    logger.info(f"Training job {job_id} started")
    return {"job_id": job_id, "status": "running", "message": f"Training started — poll /train/status/{job_id}"}


@app.get("/train/status/{job_id}")
def training_status(job_id: str):
    job = _training_jobs.get(job_id)
    if job is None:
        job_file = os.path.join(JOB_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file) as f:
                job = json.load(f)
            _training_jobs[job_id] = job
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.post("/quality-gate")
def run_quality_gate():
    """Runs pytest train-api/tests/test_model_quality.py — 422 if it fails (blocks deployment)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_model_quality.py", "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".", timeout=120,
        env={**os.environ, "ARTIFACTS_PATH": artifacts.ARTIFACTS_PATH},
    )
    passed = proc.returncode == 0
    logger.info(f"Quality gate {'PASSED' if passed else 'FAILED'} (rc={proc.returncode})")
    if not passed:
        raise HTTPException(status_code=422, detail={"status": "failed", "output": proc.stdout[-3000:]})
    return {"status": "passed", "output": proc.stdout[-3000:]}


@app.get("/health")
def health():
    running = sum(1 for j in _training_jobs.values() if j.get("status") == "running")
    return {"status": "healthy", "service": "train-api", "version": "1.0", "active_jobs": running}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
