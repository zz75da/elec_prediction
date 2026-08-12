"""Unit tests for train-api/services/artifacts.py — save/load roundtrip, no statsmodels dependency."""
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
        models={"holt_winters": {"dummy": "hw"}, "sarima": {"dummy": "sarima"}},
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
        models={"holt_winters": {"model": "hw-object"}, "sarima": {"model": "sarima-object"}},
        metadata=metadata,
        history={"best_model": "holt_winters", "metrics": {}},
    )

    model, model_type = artifacts.load_deployment_model()
    assert model_type == "holt_winters"
    assert model == {"model": "hw-object"}


def test_load_deployment_model_picks_ml_global(tmp_path):
    """The generic dict-based dispatch must work for any model name, not just the original two."""
    metadata = {"best_model": "ml_global", "metrics": {}}
    artifacts.save_artifacts(
        ols_results={"dummy": "ols"},
        models={"holt_winters": {"model": "hw"}, "sarima": {"model": "sarima"}, "ml_global": {"model": "lgbm"}},
        metadata=metadata,
        history={"best_model": "ml_global", "metrics": {}},
    )

    model, model_type = artifacts.load_deployment_model()
    assert model_type == "ml_global"
    assert model == {"model": "lgbm"}


def test_save_artifacts_skips_none_model(tmp_path):
    """A candidate that didn't run this training pass (e.g. ml_global below
    min_countries_required) is passed as None and must not produce a pickle file."""
    artifacts.save_artifacts(
        ols_results={"dummy": "ols"},
        models={"holt_winters": {"model": "hw"}, "sarima": {"model": "sarima"}, "ml_global": None},
        metadata={"best_model": "sarima", "metrics": {}},
        history={},
    )
    assert not (tmp_path / "france" / "ml_global_model.pkl").exists()
    assert (tmp_path / "france" / "sarima_model.pkl").exists()


def test_sha256_sidecar_written(tmp_path):
    artifacts.save_artifacts(
        ols_results={"a": 1}, models={"holt_winters": {"b": 2}, "sarima": {"c": 3}},
        metadata={"best_model": "sarima", "metrics": {}}, history={},
    )
    assert (tmp_path / "france" / "sarima_model.pkl.sha256").exists()


def test_load_metadata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.load_metadata()


def test_save_and_load_roundtrip_multiple_countries(tmp_path):
    """The core namespace-isolation guarantee: two countries' artifacts never collide."""
    artifacts.save_artifacts(
        ols_results={"dummy": "ols-fr"},
        models={"holt_winters": {"dummy": "hw-fr"}, "sarima": {"dummy": "sarima-fr"}},
        metadata={"best_model": "sarima", "metrics": {"sarima": {"mape": 2.22}}},
        history={"best_model": "sarima", "metrics": {}}, country="france",
    )
    artifacts.save_artifacts(
        ols_results={"dummy": "ols-usa"},
        models={"holt_winters": {"dummy": "hw-usa"}, "sarima": {"dummy": "sarima-usa"}},
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
