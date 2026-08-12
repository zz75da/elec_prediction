"""Unit tests for train-api/services/preprocess.py — regression + deseasonalization."""
import numpy as np

from services.preprocess import deseasonalize, fit_temperature_regression, run_preprocessing


def test_fit_temperature_regression_shapes(synthetic_monthly_series):
    ols_results, df = fit_temperature_regression(synthetic_monthly_series)
    assert "Conso_correction" in df.columns
    assert len(df) == len(synthetic_monthly_series)
    assert not df["Conso_correction"].isna().any()


def test_fit_temperature_regression_positive_dju_coefficient(synthetic_monthly_series):
    """Consumption rises with DJU by construction — the fitted coefficient must be positive."""
    ols_results, _ = fit_temperature_regression(synthetic_monthly_series)
    assert ols_results.params["DJU"] > 0


def test_fit_temperature_regression_good_fit(synthetic_monthly_series):
    """Synthetic data is DJU + small noise — R2 should be high."""
    ols_results, _ = fit_temperature_regression(synthetic_monthly_series)
    assert ols_results.rsquared > 0.8


def test_deseasonalize_adds_columns(synthetic_monthly_series):
    _, df = fit_temperature_regression(synthetic_monthly_series)
    df = deseasonalize(df, seasonal_periods=12)
    assert {"seasonal", "trend", "CVS"}.issubset(df.columns)


def test_run_preprocessing_stats(synthetic_monthly_series):
    df, ols_results, stats = run_preprocessing(synthetic_monthly_series, seasonal_periods=12)
    assert stats["n_observations"] == len(synthetic_monthly_series)
    assert 0 <= stats["ols_r2"] <= 1
    assert np.isfinite(stats["ols_coef_dju"])
