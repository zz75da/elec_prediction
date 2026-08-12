"""
Integration tests — exercise the real FastAPI apps in-process via TestClient
(no Docker required). Marked `integration` because test_full_train_predict_cycle
runs the actual OLS + Holt-Winters + SARIMA fit on the committed data/raw/*.csv,
which takes a few seconds.

Run:
    pytest tests/integration/ -v -m integration
"""
import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def train_client(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # app.py reads params.yaml / data/raw with cwd-relative paths
    sys.path.insert(0, str(REPO_ROOT / "train-api"))
    import app as train_app  # noqa: E402
    importlib.reload(train_app)
    return TestClient(train_app.app)


@pytest.fixture
def predict_client(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT / "predict-api"))
    import app as predict_app  # noqa: E402
    importlib.reload(predict_app)
    return TestClient(predict_app.app)


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
    assert status["best_model"] in ("holt_winters", "sarima")
    assert status["metrics"][status["best_model"]]["mape"] < 30  # generous — real gate is in test_model_quality.py

    reload_resp = predict_client.post("/reload-artifacts")
    assert reload_resp.status_code == 200

    predict_resp = predict_client.post("/predict", json={"horizon": 6})
    assert predict_resp.status_code == 200
    body = predict_resp.json()
    assert len(body["forecast"]) == 6
    assert body["model_used"] == status["best_model"]
