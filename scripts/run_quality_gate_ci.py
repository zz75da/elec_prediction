"""
Runs the training pipeline in-process (no FastAPI server needed) and then the
pytest quality gate — used by CI so it doesn't need to boot uvicorn + poll a
job endpoint just to get artifacts on disk.

Mirrors train-api/app.py's _run_training_pipeline() exactly, minus the
threading/Prometheus/job-registry plumbing that only makes sense behind a
running service.

Run from the repo root:
    python scripts/run_quality_gate_ci.py [--country france]
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train-api"))

from services import artifacts, config, data_loader, evaluate, preprocess, trainer_holtwinters, trainer_ml, trainer_sarima  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="france")
    args = parser.parse_args()
    country = args.country

    os.chdir(REPO_ROOT)
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cfg = config.resolve_country_config(params, country)

    seasonal_periods = params["preprocess"]["seasonal_periods"]
    test_year = cfg["test_year"]

    df_merged = data_loader.load_merged(cfg["raw_conso_path"], cfg["raw_dju_path"], country=country)
    df, ols_results, stats = preprocess.run_preprocessing(df_merged, seasonal_periods=seasonal_periods)
    os.makedirs(os.path.dirname(cfg["processed_path"]), exist_ok=True)
    df.to_csv(cfg["processed_path"])

    train_df = df[df.index.year < test_year]
    test_df = df[df.index.year == test_year]
    horizon = len(test_df)

    hw_cfg = params["model_holt_winters"]
    hw_model_bt = trainer_holtwinters.fit_holt_winters(
        train_df["Conso_correction"], seasonal_periods=seasonal_periods, trend=hw_cfg["trend"], seasonal=hw_cfg["seasonal"],
    )
    hw_pred = trainer_holtwinters.forecast(hw_model_bt, horizon)
    hw_metrics = evaluate.evaluate_forecast(test_df["Conso_correction"].values, hw_pred)

    sarima_cfg = params["model_sarima"]
    sarima_results_bt = trainer_sarima.fit_sarima(
        train_df["Conso_correction"], order=sarima_cfg["order"],
        seasonal_order=sarima_cfg["seasonal_order"], log_transform=sarima_cfg["log_transform"],
    )
    sarima_pred, _, _ = trainer_sarima.forecast(sarima_results_bt, horizon, log_transform=sarima_cfg["log_transform"])
    sarima_metrics = evaluate.evaluate_forecast(test_df["Conso_correction"].values, sarima_pred)

    metrics = {"holt_winters": hw_metrics, "sarima": sarima_metrics}

    ml_cfg = params["model_ml_global"]
    ml_fcst_full = None
    ml_ran = False
    if ml_cfg["enabled"]:
        other_frames = trainer_ml.load_other_country_frames(params, seasonal_periods, exclude=country)
        n_available = len(other_frames) + 1
        if n_available >= ml_cfg["min_countries_required"]:
            cutoff_years = {c: params["data"]["countries"][c]["test_year"] for c in params["data"]["countries"]}
            pooled_bt = trainer_ml.build_pooled_frame({**other_frames, country: df}, cutoff_years=cutoff_years)
            ml_fcst_bt = trainer_ml.fit_global_model(pooled_bt, ml_cfg)
            ml_pred = trainer_ml.forecast_country(ml_fcst_bt, country, horizon)
            metrics["ml_global"] = evaluate.evaluate_forecast(test_df["Conso_correction"].values, ml_pred)
            ml_ran = True

    best_model = min(metrics, key=lambda k: metrics[k]["mape"])

    hw_model_full = trainer_holtwinters.fit_holt_winters(
        df["Conso_correction"], seasonal_periods=seasonal_periods, trend=hw_cfg["trend"], seasonal=hw_cfg["seasonal"],
    )
    sarima_results_full = trainer_sarima.fit_sarima(
        df["Conso_correction"], order=sarima_cfg["order"],
        seasonal_order=sarima_cfg["seasonal_order"], log_transform=sarima_cfg["log_transform"],
    )
    if ml_ran:
        pooled_full = trainer_ml.build_pooled_frame({**other_frames, country: df}, cutoff_years=None)
        ml_fcst_full = trainer_ml.fit_global_model(pooled_full, ml_cfg)

    metadata = {
        "country": country,
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
        "country": country,
        "best_model": best_model,
        "metrics": metrics,
        "quality_gate": {
            "mape_threshold": params["quality_gate"]["mape_threshold"],
            "selected_mape": metrics[best_model]["mape"],
            "passes": metrics[best_model]["mape"] < params["quality_gate"]["mape_threshold"],
        },
    }

    artifacts.save_artifacts(
        ols_results,
        {"holt_winters": hw_model_full, "sarima": sarima_results_full, "ml_global": ml_fcst_full},
        metadata, history, country=country,
    )

    print(f"Country: {country}")
    print(f"Best model: {best_model}")
    for name, m in metrics.items():
        print(f"{name:15s} MAPE: {m['mape']}%  RMSE: {m['rmse']}")


if __name__ == "__main__":
    main()
