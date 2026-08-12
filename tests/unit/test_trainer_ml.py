"""Unit tests for train-api/services/trainer_ml.py — the pooled multi-country LightGBM
candidate. Uses small synthetic multi-country panels, no real data/network needed."""
import numpy as np
import pandas as pd
import pytest

from services import trainer_ml

ML_CFG = {
    "freq": "MS",
    "lags": [1, 2, 3],
    "date_features": ["month"],
    "static_features": ["country"],
    "lightgbm_params": {
        "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3, "num_leaves": 7,
        "min_child_samples": 3, "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbosity": -1,
    },
}


def _synthetic_country_frame(base: float, n_months: int = 36, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    seasonal = 2000 * np.cos((months.month - 1) / 12 * 2 * np.pi)
    conso_correction = base + seasonal + rng.normal(0, 100, n_months)
    return pd.DataFrame({"Conso_correction": conso_correction}, index=months)


def _pooled_country_frames():
    return {
        "france": _synthetic_country_frame(40000, seed=1),
        "usa": _synthetic_country_frame(300000, seed=2),
        "germany": _synthetic_country_frame(45000, seed=3),
    }


def test_build_pooled_frame_shape_and_columns():
    pooled = trainer_ml.build_pooled_frame(_pooled_country_frames())
    assert list(pooled.columns) == ["unique_id", "ds", "y", "country"]
    assert set(pooled["unique_id"].unique()) == {"france", "usa", "germany"}
    assert len(pooled) == 36 * 3
    assert str(pooled["country"].dtype) == "category"


def test_build_pooled_frame_cutoff_years_truncates_per_country():
    frames = _pooled_country_frames()
    pooled = trainer_ml.build_pooled_frame(frames, cutoff_years={"france": 2021, "usa": 2022})
    france_rows = pooled[pooled["unique_id"] == "france"]
    usa_rows = pooled[pooled["unique_id"] == "usa"]
    germany_rows = pooled[pooled["unique_id"] == "germany"]  # no cutoff entry -> full history
    assert france_rows["ds"].dt.year.max() == 2020
    assert usa_rows["ds"].dt.year.max() == 2021
    assert germany_rows["ds"].dt.year.max() == 2022


def test_build_pooled_frame_empty_input():
    pooled = trainer_ml.build_pooled_frame({})
    assert pooled.empty
    assert list(pooled.columns) == ["unique_id", "ds", "y", "country"]


def test_fit_and_forecast_country_roundtrip():
    pooled = trainer_ml.build_pooled_frame(_pooled_country_frames())
    fcst = trainer_ml.fit_global_model(pooled, ML_CFG)

    horizon = 6
    pred = trainer_ml.forecast_country(fcst, "france", horizon)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == horizon
    assert np.all(np.isfinite(pred))
    # Forecasts should be in the right ballpark for the series they came from, not e.g.
    # accidentally returning another country's much larger-scale predictions.
    assert 20000 < pred.mean() < 60000


def test_forecast_country_is_series_specific():
    pooled = trainer_ml.build_pooled_frame(_pooled_country_frames())
    fcst = trainer_ml.fit_global_model(pooled, ML_CFG)
    france_pred = trainer_ml.forecast_country(fcst, "france", 3)
    usa_pred = trainer_ml.forecast_country(fcst, "usa", 3)
    assert usa_pred.mean() > france_pred.mean() * 2  # usa's base (300k) vs france's (40k)


def test_load_other_country_frames_skips_bad_country(monkeypatch):
    """A country whose data fails to load must be skipped (logged), not raise."""
    params = {
        "data": {
            "countries": {
                "france": {"raw_conso_path": "x", "raw_dju_path": "y"},
                "atlantis": {"raw_conso_path": "missing.csv", "raw_dju_path": "missing2.csv"},
            }
        }
    }

    def fake_load_merged(conso_path, dju_path, country):
        if country == "atlantis":
            raise FileNotFoundError("no such file")
        return pd.DataFrame({"Consommation": [1.0] * 24, "DJU": [1.0] * 24},
                             index=pd.date_range("2020-01-01", periods=24, freq="MS"))

    def fake_run_preprocessing(df_merged, seasonal_periods):
        df = df_merged.copy()
        df["Conso_correction"] = df["Consommation"]
        return df, None, {}

    monkeypatch.setattr(trainer_ml.data_loader, "load_merged", fake_load_merged)
    monkeypatch.setattr(trainer_ml.preprocess, "run_preprocessing", fake_run_preprocessing)

    frames = trainer_ml.load_other_country_frames(params, seasonal_periods=12, exclude="usa")
    assert "france" in frames
    assert "atlantis" not in frames  # skipped, not raised
