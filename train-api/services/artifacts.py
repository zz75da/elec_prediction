# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : sauvegarde/chargement atomique des artefacts produits
# par un run d'entrainement (modeles pickle, metadonnees JSON,
# historique pour le quality gate). Pattern repris de
# rakuten_mlops_services/train-api/services/artifacts.py
# (ecriture via fichier temporaire + os.replace + empreinte SHA256).
#
# Fonctions principales :
#   - save_artifacts(ols_results, hw_model, sarima_results, metadata,
#     skip_existing=False) : ecrit dans ARTIFACTS_PATH
#     ols_model.pkl, holt_winters_model.pkl, sarima_model.pkl,
#     model_metadata.json, train_history.json (+ .sha256 pour les pickles)
#   - load_deployment_model() -> objet modele + type ("holt_winters"|"sarima")
#     choisi via model_metadata.json["best_model"]
#   - load_metadata() -> dict
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
) -> None:
    """
    Persist every artifact from a training run.

    metadata must contain at least:
      {"best_model": "holt_winters"|"sarima", "sarima_log_transform": bool,
       "metrics": {...}, "params": {...}}
    history is the flat dict written to train_history.json for the quality gate.
    """
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    t0 = time()

    _atomic_pickle_dump(ols_results, os.path.join(ARTIFACTS_PATH, "ols_model.pkl"))
    _atomic_pickle_dump(hw_model, os.path.join(ARTIFACTS_PATH, "holt_winters_model.pkl"))
    _atomic_pickle_dump(sarima_results, os.path.join(ARTIFACTS_PATH, "sarima_model.pkl"))

    with open(os.path.join(ARTIFACTS_PATH, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(ARTIFACTS_PATH, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"[Artifacts] Saved ols/holt_winters/sarima models + metadata in {time() - t0:.2f}s -> {ARTIFACTS_PATH}")


def load_metadata() -> dict:
    path = os.path.join(ARTIFACTS_PATH, "model_metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model_metadata.json found in {ARTIFACTS_PATH} — run /train first")
    with open(path) as f:
        return json.load(f)


def load_history() -> dict:
    path = os.path.join(ARTIFACTS_PATH, "train_history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No train_history.json found in {ARTIFACTS_PATH} — run /train first")
    with open(path) as f:
        return json.load(f)


def _load_pickle(name: str):
    path = os.path.join(ARTIFACTS_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_ols_model():
    return _load_pickle("ols_model.pkl")


def load_deployment_model():
    """
    Returns (model_object, model_type) for whichever model was flagged
    best_model in model_metadata.json — this is what predict-api serves.
    """
    metadata = load_metadata()
    best = metadata.get("best_model", "sarima")
    if best == "holt_winters":
        return _load_pickle("holt_winters_model.pkl"), "holt_winters"
    return _load_pickle("sarima_model.pkl"), "sarima"
