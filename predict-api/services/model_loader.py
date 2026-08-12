# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : chargement (lecture seule) des artefacts ecrits par
# train-api/services/artifacts.py sur le volume partage
# data/artifacts, namespaces par pays sous ARTIFACTS_PATH/<country>/.
# Duplique volontairement le loader plutot que de partager un
# package Python entre les deux services — meme principe
# d'independance des microservices que rakuten_mlops_services
# (predict-api reimplemente son propre chargement plutot que
# d'importer train-api).
#
# Fonctions principales :
#   - load_metadata(country) -> dict (model_metadata.json)
#   - load_deployment_model(country) -> (model, model_type, metadata)
#   - load_ols_model(country) -> objet statsmodels OLSResults
#   - list_countries() -> list[str] : sous-dossiers d'ARTIFACTS_PATH
#     contenant un model_metadata.json (utilise au demarrage / par
#     /reload-artifacts sans pays explicite pour recharger tout ce
#     qui est disponible sur disque)
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


def _country_dir(country: str) -> str:
    return os.path.join(ARTIFACTS_PATH, country)


def _load_pickle(name: str, country: str = "france"):
    path = os.path.join(_country_dir(country), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path} — has train-api completed a /train run?")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(country: str = "france") -> dict:
    path = os.path.join(_country_dir(country), "model_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model_metadata.json in {_country_dir(country)} — has train-api completed a /train run?")
    with open(path) as f:
        return json.load(f)


def load_deployment_model(country: str = "france"):
    metadata = load_metadata(country)
    best = metadata.get("best_model", "sarima")
    return _load_pickle(f"{best}_model.pkl", country), best, metadata


def load_ols_model(country: str = "france"):
    return _load_pickle("ols_model.pkl", country)


def list_countries() -> list:
    if not os.path.isdir(ARTIFACTS_PATH):
        return []
    return sorted(
        name for name in os.listdir(ARTIFACTS_PATH)
        if os.path.isfile(os.path.join(ARTIFACTS_PATH, name, "model_metadata.json"))
    )
