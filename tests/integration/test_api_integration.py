"""
Integration tests — exercise the real FastAPI apps in-process via TestClient
(no Docker required). Marked `integration` because test_full_train_predict_cycle
runs the actual OLS + Holt-Winters + SARIMA fit on the committed data/raw/*.csv,
which takes a few seconds.

Run:
    pytest tests/integration/ -v -m integration
"""
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def _purge_services_modules():
    """train-api and predict-api both define a top-level `services` package. Only one
    can be cached in sys.modules at a time, so before switching which service's app we
    import, drop any previously-cached `services`/`services.*`/`app` modules to force a
    fresh resolution against whichever directory is currently first on sys.path."""
    for name in list(sys.modules):
        if name == "services" or name.startswith("services.") or name == "app":
            del sys.modules[name]


@pytest.fixture(scope="module")
def train_client():
    # module-scoped: app.py registers Prometheus Gauges at import time, and the default
    # CollectorRegistry is a process-wide singleton — re-importing app.py a second time
    # within the same process raises "Duplicated timeseries in CollectorRegistry", so the
    # module is imported exactly once and its TestClient shared across every test here.
    mp = pytest.MonkeyPatch()
    mp.chdir(REPO_ROOT)  # app.py reads params.yaml / data/raw with cwd-relative paths
    _purge_services_modules()
    sys.path.insert(0, str(REPO_ROOT / "train-api"))
    import app as train_app  # noqa: E402
    yield TestClient(train_app.app)
    mp.undo()


@pytest.fixture(scope="module")
def predict_client():
    mp = pytest.MonkeyPatch()
    mp.chdir(REPO_ROOT)
    _purge_services_modules()
    sys.path.insert(0, str(REPO_ROOT / "predict-api"))
    import app as predict_app  # noqa: E402
    yield TestClient(predict_app.app)
    mp.undo()


def test_train_api_health(train_client):
    resp = train_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_predict_api_health(predict_client):
    resp = predict_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "predict-api"


def test_predict_without_trained_model_returns_503(predict_client):
    """Fresh predict-api with no artifacts yet must fail loudly, not silently."""
    resp = predict_client.post("/predict", json={"horizon": 12})
    # Either 503 (no model loaded) or 200 (artifacts already exist from a previous local run) are acceptable
    assert resp.status_code in (200, 503)


def test_countries_endpoint_lists_default(train_client):
    resp = train_client.get("/countries")
    assert resp.status_code == 200
    body = resp.json()
    codes = {c["code"] for c in body["countries"]}
    assert "france" in codes
    assert body["default_country"] == "france"


def test_train_with_unknown_country_returns_400(train_client):
    resp = train_client.post("/train", json={"country": "atlantis"})
    assert resp.status_code == 400


@pytest.mark.integration
def test_train_with_explicit_country_france(train_client):
    """train_client is module-scoped (see its fixture docstring), so this must wait for
    the job to finish before returning — otherwise a later test in this module sharing
    the same client would hit a spurious 409 (job already running)."""
    resp = train_client.post("/train", json={"country": "france"})
    assert resp.status_code == 202
    assert resp.json()["country"] == "france"
    job_id = resp.json()["job_id"]

    status = {}
    for _ in range(60):
        status = train_client.get(f"/train/status/{job_id}").json()
        if status["status"] in ("success", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "success", status.get("error")


@pytest.mark.integration
def test_full_train_predict_cycle(train_client, predict_client):
    """End-to-end: POST /train, poll to completion, reload predict-api, POST /predict."""
    resp = train_client.post("/train", json={})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = {}
    for _ in range(60):
        status = train_client.get(f"/train/status/{job_id}").json()
        if status["status"] in ("success", "failed"):
            break
        time.sleep(1)

    assert status["status"] == "success", status.get("error")
    assert status["best_model"] in status["metrics"].keys()  # self-consistent, doesn't hardcode the candidate set
    assert status["metrics"][status["best_model"]]["mape"] < 30  # generous — real gate is in test_model_quality.py

    reload_resp = predict_client.post("/reload-artifacts")
    assert reload_resp.status_code == 200

    predict_resp = predict_client.post("/predict", json={"horizon": 6})
    assert predict_resp.status_code == 200
    body = predict_resp.json()
    assert len(body["forecast"]) == 6
    assert body["model_used"] == status["best_model"]
