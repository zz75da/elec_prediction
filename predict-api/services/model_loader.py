# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : chargement (lecture seule) des artefacts ecrits par
# train-api/services/artifacts.py sur le volume partage
# data/artifacts. Duplique volontairement le loader plutot que
# de partager un package Python entre les deux services —
# meme principe d'independance des microservices que
# rakuten_mlops_services (predict-api reimplemente son propre
# chargement plutot que d'importer train-api).
#
# Fonctions principales :
#   - load_metadata() -> dict (model_metadata.json)
#   - load_deployment_model() -> (model, model_type)
#   - load_ols_model() -> objet statsmodels OLSResults
#
# Variables / constantes importantes :
#   - ARTIFACTS_PATH (env ARTIFACTS_PATH, def. "data/artifacts")
#
# Dependances externes : pickle, json (stdlib)
# ============================================================
import json
import os
import pickle

ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "data/artifacts")


def _load_pickle(name: str):
    path = os.path.join(ARTIFACTS_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path} — has train-api completed a /train run?")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata() -> dict:
    path = os.path.join(ARTIFACTS_PATH, "model_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model_metadata.json in {ARTIFACTS_PATH} — has train-api completed a /train run?")
    with open(path) as f:
        return json.load(f)


def load_deployment_model():
    metadata = load_metadata()
    best = metadata.get("best_model", "sarima")
    if best == "holt_winters":
        return _load_pickle("holt_winters_model.pkl"), "holt_winters", metadata
    return _load_pickle("sarima_model.pkl"), "sarima", metadata


def load_ols_model():
    return _load_pickle("ols_model.pkl")
