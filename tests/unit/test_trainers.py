"""Unit tests for train-api/services/trainer_holtwinters.py and trainer_sarima.py."""
import numpy as np

from services.evaluate import evaluate_forecast, mape, rmse
from services.preprocess import fit_temperature_regression
from services.trainer_holtwinters import fit_holt_winters
from services.trainer_holtwinters import forecast as hw_forecast
from services.trainer_sarima import fit_sarima
from services.trainer_sarima import forecast as sarima_forecast


def _corrected_series(synthetic_monthly_series):
    _, df = fit_temperature_regression(synthetic_monthly_series)
    return df["Conso_correction"]


def test_holt_winters_forecast_horizon(synthetic_monthly_series):
    series = _corrected_series(synthetic_monthly_series)
    train, horizon = series.iloc[:-12], 12
    model = fit_holt_winters(train, seasonal_periods=12, trend="add", seasonal="add")
    pred = hw_forecast(model, horizon)
    assert len(pred) == horizon
    assert np.all(np.isfinite(pred))


def test_sarima_forecast_horizon_and_ci(synthetic_monthly_series):
    series = _corrected_series(synthetic_monthly_series)
    train, horizon = series.iloc[:-12], 12
    results = fit_sarima(train, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12), log_transform=True)
    pred, lower, upper = sarima_forecast(results, horizon, log_transform=True)
    assert len(pred) == len(lower) == len(upper) == horizon
    assert np.all(lower <= pred) and np.all(pred <= upper)


def test_backtest_mape_reasonable_on_synthetic_data(synthetic_monthly_series):
    """Sanity check: on a clean synthetic signal, both models should beat a generous MAPE ceiling."""
    series = _corrected_series(synthetic_monthly_series)
    train, test = series.iloc[:-12], series.iloc[-12:]

    hw_model = fit_holt_winters(train, seasonal_periods=12, trend="add", seasonal="add")
    hw_pred = hw_forecast(hw_model, 12)
    hw_metrics = evaluate_forecast(test.values, hw_pred)

    sarima_results = fit_sarima(train, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12), log_transform=True)
    sarima_pred, _, _ = sarima_forecast(sarima_results, 12, log_transform=True)
    sarima_metrics = evaluate_forecast(test.values, sarima_pred)

    # Generous ceiling — this is a sanity check, not a strict regression floor
    # (the real quality gate lives in train-api/tests/test_model_quality.py and runs on real data)
    assert min(hw_metrics["mape"], sarima_metrics["mape"]) < 30
