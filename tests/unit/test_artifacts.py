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
    assert (tmp_path / "sarima_model.pkl.sha256").exists()


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.load_metadata()
