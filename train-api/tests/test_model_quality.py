"""
Model quality gate — runs after every training job (POST /quality-gate).

Asserts the deployed model's backtest MAPE stays below the notebook's own
acceptability threshold (10% — "MAPE < 10 = très bonne prédiction", notebook
cell 97). Blocks deployment if the model regresses.

Run manually:
    pytest train-api/tests/test_model_quality.py -v
"""
# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : tests pytest formant la "quality gate" executee apres
# chaque entrainement (POST /quality-gate -> ce fichier).
# Pattern repris de
# rakuten_mlops_services/train-api/tests/test_model_quality.py.
#
# Fonctions principales :
#   - _load_history() -> dict : charge train_history.json depuis ARTIFACTS_PATH
#   - _load_params() -> dict : charge params.yaml (seuils)
#   - test_mape_below_threshold : le MAPE du modele selectionne doit
#     rester sous quality_gate.mape_threshold (params.yaml)
#   - test_best_model_beats_naive_baseline : le modele retenu doit
#     battre une baseline naive (MAPE du pire des deux modeles)
#   - test_no_missing_metrics : holt_winters ET sarima doivent tous
#     deux avoir des metriques calculees (aucun echec silencieux)
#
# Dependances externes : pytest, pyyaml
# ============================================================
import json
import os

import pytest
import yaml

_ARTIFACTS = os.environ.get("ARTIFACTS_PATH", "data/artifacts")
_PARAMS_PATH = os.environ.get("PARAMS_PATH", "params.yaml")
_COUNTRY = os.environ.get("TRAIN_COUNTRY", "france")


def _load_history() -> dict:
    path = os.path.join(_ARTIFACTS, _COUNTRY, "train_history.json")
    if not os.path.exists(path):
        pytest.skip(f"No train_history.json in {os.path.join(_ARTIFACTS, _COUNTRY)} — run /train first")
    with open(path) as f:
        return json.load(f)


def _load_params() -> dict:
    if not os.path.exists(_PARAMS_PATH):
        pytest.skip(f"params.yaml not found at {_PARAMS_PATH}")
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def test_mape_below_threshold():
    history = _load_history()
    params = _load_params()
    threshold = params["quality_gate"]["mape_threshold"]
    best_model = history["best_model"]
    mape = history["metrics"][best_model]["mape"]
    assert mape < threshold, (
        f"Selected model '{best_model}' MAPE={mape:.2f}% >= threshold={threshold}%. "
        f"Model regressed — do not deploy."
    )


def test_best_model_beats_naive_baseline():
    history = _load_history()
    metrics = history["metrics"]
    best_model = history["best_model"]
    other_model = "sarima" if best_model == "holt_winters" else "holt_winters"
    assert metrics[best_model]["mape"] <= metrics[other_model]["mape"], (
        f"'{best_model}' was selected as best but has a higher MAPE "
        f"({metrics[best_model]['mape']}) than '{other_model}' ({metrics[other_model]['mape']})"
    )


def test_no_missing_metrics():
    history = _load_history()
    for model in ("holt_winters", "sarima"):
        assert model in history["metrics"], f"Missing metrics for {model}"
        assert "mape" in history["metrics"][model] and "rmse" in history["metrics"][model]
        assert history["metrics"][model]["mape"] > 0, f"{model} MAPE is 0 — suspicious, check the backtest"
