# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : microservice FastAPI (port 5011) qui sert les previsions
# de consommation electrique a partir du modele deploye par
# train-api (Holt-Winters ou SARIMA, selon model_metadata.json),
# pour un ou plusieurs pays (ARTIFACTS_PATH/<country>/...).
# Expose /predict (horizon en mois, pays optionnel), /health,
# /metrics (Prometheus). Pattern d'architecture (chargement des
# artefacts au demarrage, Prometheus request/latence, reload a
# chaud) repris de rakuten_mlops_services/predict-api/app.py.
#
# Fonctions principales :
#   - startup_event() : charge tous les pays disponibles sur disque
#     (model_loader.list_countries()) dans _state[country]
#   - POST /reload-artifacts {country?} : recharge un pays precis
#     (ou tous les pays disponibles si omis), sans redemarrer le
#     conteneur (appele par Streamlit apres un nouvel entrainement)
#   - POST /predict {horizon, dju_forecast?, country?} : projette
#     horizon mois au-dela du dernier mois observe pour le pays
#     demande (defaut DEFAULT_COUNTRY) ; si dju_forecast est fourni
#     (meme longueur que horizon), reconstruit aussi la consommation
#     reelle prevue
#   - GET /health, GET /metrics
#
# Variables / constantes importantes :
#   - ARTIFACTS_PATH (via services.model_loader)
#   - DEFAULT_COUNTRY (env DEFAULT_COUNTRY, def. "france")
#   - Compteurs/histos Prometheus : PREDICT_REQUEST_COUNT, PREDICT_LATENCY
#
# Dependances externes : fastapi, pydantic, prometheus_client,
# prometheus_fastapi_instrumentator, numpy, pandas, services.* (interne)
# ============================================================
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from services import forecast, model_loader

app = FastAPI(title="Predict API — elec_prediction", description="Electricity consumption forecasting", version="1.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

PREDICT_REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["country"])
PREDICT_LATENCY = Histogram("predict_request_latency_seconds", "Prediction request latency", ["country"])

DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "france")

# --- Globals loaded at startup, keyed by country code ---
_state: Dict[str, dict] = {}


def _load_country(country: str):
    model, model_type, metadata = model_loader.load_deployment_model(country)
    ols_model = model_loader.load_ols_model(country)
    _state[country] = {"model": model, "model_type": model_type, "metadata": metadata, "ols_model": ols_model}


@app.on_event("startup")
def startup_event():
    for country in model_loader.list_countries():
        try:
            _load_country(country)
        except FileNotFoundError as e:
            # Shouldn't happen (list_countries only returns countries with metadata present),
            # but stay defensive — a partial artifact set shouldn't crash startup.
            print(f"[predict-api] Failed to load country={country} at startup: {e}")
    if not _state:
        # Service can start before train-api has produced any artifacts —
        # /predict will 503 until /reload-artifacts is called.
        print("[predict-api] No artifacts for any country yet at startup")


@app.post("/reload-artifacts")
def reload_artifacts(country: Optional[str] = None):
    targets = [country] if country else (model_loader.list_countries() or [DEFAULT_COUNTRY])
    reloaded = []
    errors = {}
    for c in targets:
        try:
            _load_country(c)
            reloaded.append(c)
        except FileNotFoundError as e:
            errors[c] = str(e)
    if not reloaded:
        raise HTTPException(status_code=404, detail=errors or "No artifacts found for any country")
    return {
        "status": "reloaded",
        "countries": reloaded,
        "errors": errors or None,
        "model_types": {c: _state[c]["model_type"] for c in reloaded},
    }


class PredictRequest(BaseModel):
    horizon: int = 12
    dju_forecast: Optional[List[float]] = None  # optional projected DJU per future month, to reconstruct real Consommation
    country: Optional[str] = None  # defaults to DEFAULT_COUNTRY if omitted


@app.post("/predict")
def predict(req: PredictRequest):
    country = req.country or DEFAULT_COUNTRY
    PREDICT_REQUEST_COUNT.labels(country=country).inc()
    t0 = time.time()

    country_state = _state.get(country)
    if country_state is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model loaded for country='{country}' — run POST /train on train-api "
                    f"(with {{'country': '{country}'}}), then POST /reload-artifacts here",
        )

    if req.horizon < 1 or req.horizon > 60:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 60 months")

    if req.dju_forecast is not None and len(req.dju_forecast) != req.horizon:
        raise HTTPException(status_code=400, detail="dju_forecast length must equal horizon")

    model_type = country_state["model_type"]
    metadata = country_state["metadata"]

    if model_type == "holt_winters":
        pred = forecast.forecast_holt_winters(country_state["model"], req.horizon)
        lower = upper = None
    else:
        pred, lower, upper = forecast.forecast_sarima(
            country_state["model"], req.horizon, log_transform=metadata.get("sarima_log_transform", True),
        )

    last_month = pd.Timestamp(metadata["last_observed_month"])
    future_months = pd.date_range(last_month, periods=req.horizon + 1, freq="M")[1:]

    consommation_pred = None
    if req.dju_forecast is not None:
        consommation_pred = forecast.reconstruct_consumption(pred, req.dju_forecast, country_state["ols_model"])

    forecast_rows = []
    for i, month in enumerate(future_months):
        row = {
            "month": month.strftime("%Y-%m"),
            "conso_correction_pred_gwh": round(float(pred[i]), 1),
        }
        if lower is not None:
            row["ci_lower_gwh"] = round(float(lower[i]), 1)
            row["ci_upper_gwh"] = round(float(upper[i]), 1)
        if consommation_pred is not None:
            row["consommation_pred_gwh"] = round(float(consommation_pred[i]), 1)
        forecast_rows.append(row)

    PREDICT_LATENCY.labels(country=country).observe(time.time() - t0)

    return {
        "country": country,
        "model_used": model_type,
        "last_observed_month": metadata["last_observed_month"],
        "backtest_metrics": metadata["metrics"],
        "note": "conso_correction_pred is the temperature-corrected forecast. Provide dju_forecast "
                "(one DJU value per future month) to also get consommation_pred_gwh (real forecast).",
        "forecast": forecast_rows,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "predict-api",
        "version": "1.0",
        "model_loaded_countries": sorted(_state.keys()),
        "model_types": {c: s["model_type"] for c, s in _state.items()},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5011)
