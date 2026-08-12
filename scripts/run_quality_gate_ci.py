"""
Runs the training pipeline in-process (no FastAPI server needed) and then the
pytest quality gate — used by CI so it doesn't need to boot uvicorn + poll a
job endpoint just to get artifacts on disk.

Mirrors train-api/app.py's _run_training_pipeline() exactly, minus the
threading/Prometheus/job-registry plumbing that only makes sense behind a
running service.

Run from the repo root:
    python scripts/run_quality_gate_ci.py
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train-api"))

from services import artifacts, data_loader, evaluate, preprocess, trainer_holtwinters, trainer_sarima  # noqa: E402


def main():
    os.chdir(REPO_ROOT)
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    seasonal_periods = params["preprocess"]["seasonal_periods"]
    test_year = params["data"]["test_year"]

    df_merged = data_loader.load_merged(params["data"]["raw_conso_path"], params["data"]["raw_dju_path"])
    df, ols_results, stats = preprocess.run_preprocessing(df_merged, seasonal_periods=seasonal_periods)
    os.makedirs(os.path.dirname(params["data"]["processed_path"]), exist_ok=True)
    df.to_csv(params["data"]["processed_path"])

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
    best_model = "sarima" if sarima_metrics["mape"] <= hw_metrics["mape"] else "holt_winters"

    hw_model_full = trainer_holtwinters.fit_holt_winters(
        df["Conso_correction"], seasonal_periods=seasonal_periods, trend=hw_cfg["trend"], seasonal=hw_cfg["seasonal"],
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
        "best_model": best_model,
        "metrics": metrics,
        "quality_gate": {
            "mape_threshold": params["quality_gate"]["mape_threshold"],
            "selected_mape": metrics[best_model]["mape"],
            "passes": metrics[best_model]["mape"] < params["quality_gate"]["mape_threshold"],
        },
    }

    artifacts.save_artifacts(ols_results, hw_model_full, sarima_results_full, metadata, history)

    print(f"Best model: {best_model}")
    print(f"Holt-Winters MAPE: {hw_metrics['mape']}%  RMSE: {hw_metrics['rmse']}")
    print(f"SARIMA MAPE:       {sarima_metrics['mape']}%  RMSE: {sarima_metrics['rmse']}")


if __name__ == "__main__":
    main()
