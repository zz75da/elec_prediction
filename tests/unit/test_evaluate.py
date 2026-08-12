"""Unit tests for train-api/services/evaluate.py — pure numpy, no statsmodels dependency."""
import numpy as np
import pytest

from services.evaluate import evaluate_forecast, mape, rmse


def test_rmse_zero_for_perfect_prediction():
    actual = np.array([10.0, 20.0, 30.0])
    assert rmse(actual, actual) == 0.0


def test_rmse_known_value():
    actual = np.array([0.0, 0.0])
    predicted = np.array([3.0, 4.0])
    # sqrt(mean([9, 16])) = sqrt(12.5)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))


def test_mape_zero_for_perfect_prediction():
    actual = np.array([100.0, 200.0])
    assert mape(actual, actual) == 0.0


def test_mape_known_value():
    actual = np.array([100.0])
    predicted = np.array([90.0])
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_evaluate_forecast_returns_both_metrics():
    actual = np.array([100.0, 110.0, 90.0])
    predicted = np.array([102.0, 108.0, 95.0])
    result = evaluate_forecast(actual, predicted)
    assert set(result.keys()) == {"rmse", "mape"}
    assert result["rmse"] > 0
    assert result["mape"] > 0
