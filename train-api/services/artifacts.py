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
#   - save_artifacts(ols_results, models, metadata, history,
#     country="france") : ecrit dans ARTIFACTS_PATH/<country>/ un
#     fichier ols_model.pkl + un fichier "<nom>_model.pkl" par
#     entree du dict `models` (ex: holt_winters_model.pkl,
#     sarima_model.pkl, ml_global_model.pkl), model_metadata.json,
#     train_history.json (+ .sha256 pour les pickles). Le dict
#     generique (plutot que des arguments positionnels fixes)
#     permet d'ajouter un 4e candidat sans toucher ce fichier.
#   - load_deployment_model(country) -> objet modele + type,
#     choisi via model_metadata.json["best_model"] — dispatch
#     generique par nom de fichier, pas de if/else par modele
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
from typing import Any, Dict

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
    models: Dict[str, Any],
    metadata: dict,
    history: dict,
    country: str = "france",
) -> None:
    """
    Persist every artifact from a training run, namespaced under ARTIFACTS_PATH/<country>/.

    `models` maps a model name (e.g. "holt_winters", "sarima", "ml_global") to its fitted
    object; each is written as "<name>_model.pkl". A None value is skipped (lets a candidate
    that was disabled/unavailable for this run — e.g. ml_global below min_countries_required —
    be omitted cleanly).

    metadata must contain at least:
      {"best_model": <one of models' keys>, "sarima_log_transform": bool,
       "metrics": {...}, "params": {...}}
    history is the flat dict written to train_history.json for the quality gate.
    """
    country_dir = _country_dir(country)
    os.makedirs(country_dir, exist_ok=True)
    t0 = time()

    _atomic_pickle_dump(ols_results, os.path.join(country_dir, "ols_model.pkl"))
    for name, obj in models.items():
        if obj is not None:
            _atomic_pickle_dump(obj, os.path.join(country_dir, f"{name}_model.pkl"))

    with open(os.path.join(country_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(os.path.join(country_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    saved_names = ", ".join(n for n, o in models.items() if o is not None)
    print(f"[Artifacts] Saved ols/{saved_names} models + metadata in {time() - t0:.2f}s -> {country_dir}")


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
    return _load_pickle(f"{best}_model.pkl", country), best
