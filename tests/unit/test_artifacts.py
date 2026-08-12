"""Unit tests for train-api/services/artifacts.py — save/load roundtrip, no statsmodels dependency."""
import json

import pytest

from services import artifacts


@pytest.fixture(autouse=True)
def _isolated_artifacts_path(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACTS_PATH", str(tmp_path))
    yield tmp_path


def test_save_and_load_metadata(tmp_path):
    metadata = {"best_model": "sarima", "sarima_log_transform": True, "metrics": {"sarima": {"mape": 2.22}}}
    history = {"best_model": "sarima", "metrics": {"sarima": {"mape": 2.22}, "holt_winters": {"mape": 5.0}}}

    artifacts.save_artifacts(
        ols_results={"dummy": "ols"},
        hw_model={"dummy": "hw"},
        sarima_results={"dummy": "sarima"},
        metadata=metadata,
        history=history,
    )

    loaded_meta = artifacts.load_metadata()
    assert loaded_meta["best_model"] == "sarima"

    loaded_history = artifacts.load_history()
    assert loaded_history["metrics"]["sarima"]["mape"] == 2.22


def test_load_deployment_model_picks_best(tmp_path):
    metadata = {"best_model": "holt_winters", "metrics": {}}
    artifacts.save_artifacts(
        ols_results={"dummy": "ols"},
        hw_model={"model": "hw-object"},
        sarima_results={"model": "sarima-object"},
        metadata=metadata,
        history={"best_model": "holt_winters", "metrics": {}},
    )

    model, model_type = artifacts.load_deployment_model()
    assert model_type == "holt_winters"
    assert model == {"model": "hw-object"}


def test_sha256_sidecar_written(tmp_path):
    artifacts.save_artifacts(
        ols_results={"a": 1}, hw_model={"b": 2}, sarima_results={"c": 3},
        metadata={"best_model": "sarima", "metrics": {}}, history={},
    )
    assert (tmp_path / "france" / "sarima_model.pkl.sha256").exists()


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.load_metadata()


def test_save_and_load_roundtrip_multiple_countries(tmp_path):
    """The core namespace-isolation guarantee: two countries' artifacts never collide."""
    artifacts.save_artifacts(
        ols_results={"dummy": "ols-fr"}, hw_model={"dummy": "hw-fr"}, sarima_results={"dummy": "sarima-fr"},
        metadata={"best_model": "sarima", "metrics": {"sarima": {"mape": 2.22}}},
        history={"best_model": "sarima", "metrics": {}}, country="france",
    )
    artifacts.save_artifacts(
        ols_results={"dummy": "ols-usa"}, hw_model={"dummy": "hw-usa"}, sarima_results={"dummy": "sarima-usa"},
        metadata={"best_model": "holt_winters", "metrics": {"holt_winters": {"mape": 5.0}}},
        history={"best_model": "holt_winters", "metrics": {}}, country="usa",
    )

    france_meta = artifacts.load_metadata(country="france")
    usa_meta = artifacts.load_metadata(country="usa")
    assert france_meta["best_model"] == "sarima"
    assert usa_meta["best_model"] == "holt_winters"

    france_model, france_type = artifacts.load_deployment_model(country="france")
    usa_model, usa_type = artifacts.load_deployment_model(country="usa")
    assert france_type == "sarima" and france_model == {"dummy": "sarima-fr"}
    assert usa_type == "holt_winters" and usa_model == {"dummy": "hw-usa"}
