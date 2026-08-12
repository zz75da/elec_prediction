# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : sauvegarde/chargement atomique des artefacts produits
# par un run d'entrainement (modeles pickle, metadonnees JSON,
# historique pour le quality gate), namespaces par pays sous
# ARTIFACTS_PATH/<country>/. Pattern repris de
# rakuten_mlops_services/train-api/services/artifacts.py
# (ecriture via fichier temporaire + os.replace + empreinte SHA256).
#
# Fonctions principales :
#   - save_artifacts(ols_results, hw_model, sarima_results, metadata,
#     history, country="france") : ecrit dans ARTIFACTS_PATH/<country>/
#     ols_model.pkl, holt_winters_model.pkl, sarima_model.pkl,
#     model_metadata.json, train_history.json (+ .sha256 pour les pickles)
#   - load_deployment_model(country) -> objet modele + type ("holt_winters"|"sarima")
#     choisi via model_metadata.json["best_model"]
#   - load_metadata(country) -> dict
#
# Variables / constantes importantes :
#   - ARTIFACTS_PATH (env ARTIFACTS_PATH, def. "data/artifacts")
#
# Dependances externes : pickle, hashlib, tempfile (stdlib)
# ============================================================
import hashlib
import json
import os
import pickle
import tempfile
from time import time

ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "data/artifacts")


def _country_dir(country: str) -> str:
    return os.path.join(ARTIFACTS_PATH, country)


def _atomic_pickle_dump(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), suffix=".tmp", delete=False) as tf:
        pickle.dump(obj, tf)
        tmp_path = tf.name
    os.replace(tmp_path, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(path + ".sha256", "w") as hf:
        hf.write(digest)


def save_artifacts(
    ols_results,
    hw_model,
    sarima_results,
    metadata: dict,
    history: dict,
    country: str = "france",
) -> None:
    """
    Persist every artifact from a training run, namespaced under ARTIFACTS_PATH/<country>/.

    metadata must contain at least:
      {"best_model": "holt_winters"|"sarima", "sarima_log_transform": bool,
       "metrics": {...}, "params": {...}}
    history is the flat dict written to train_history.json for the quality gate.
    """
    country_dir = _country_dir(country)
    os.makedirs(country_dir, exist_ok=True)
    t0 = time()

    _atomic_pickle_dump(ols_results, os.path.join(country_dir, "ols_model.pkl"))
    _atomic_pickle_dump(hw_model, os.path.join(country_dir, "holt_winters_model.pkl"))
    _atomic_pickle_dump(sarima_results, os.path.join(country_dir, "sarima_model.pkl"))

    with open(os.path.join(country_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(country_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"[Artifacts] Saved ols/holt_winters/sarima models + metadata in {time() - t0:.2f}s -> {country_dir}")


def load_metadata(country: str = "france") -> dict:
    path = os.path.join(_country_dir(country), "model_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model_metadata.json found in {_country_dir(country)} — run /train first")
    with open(path) as f:
        return json.load(f)


def load_history(country: str = "france") -> dict:
    path = os.path.join(_country_dir(country), "train_history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No train_history.json found in {_country_dir(country)} — run /train first")
    with open(path) as f:
        return json.load(f)


def _load_pickle(name: str, country: str = "france"):
    path = os.path.join(_country_dir(country), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ols_model(country: str = "france"):
    return _load_pickle("ols_model.pkl", country)


def load_deployment_model(country: str = "france"):
    """
    Returns (model_object, model_type) for whichever model was flagged
    best_model in model_metadata.json — this is what predict-api serves.
    """
    metadata = load_metadata(country)
    best = metadata.get("best_model", "sarima")
    if best == "holt_winters":
        return _load_pickle("holt_winters_model.pkl", country), "holt_winters"
    return _load_pickle("sarima_model.pkl", country), "sarima"
