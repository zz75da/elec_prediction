# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : microservice FastAPI (port 5011) qui sert les previsions
# de consommation electrique a partir du modele deploye par
# train-api (Holt-Winters ou SARIMA, selon model_metadata.json).
# Expose /predict (horizon en mois), /health, /metrics (Prometheus).
# Pattern d'architecture (chargement des artefacts au demarrage,
# Prometheus request/latence, reload a chaud) repris de
# rakuten_mlops_services/predict-api/app.py.
#
# Fonctions principales :
#   - startup_event() : charge le modele de deploiement + le modele
#     OLS au demarrage (variables globales model, model_type, ols_model)
#   - POST /reload-artifacts : recharge les artefacts sans redemarrer
#     le conteneur (appele par train-api ou manuellement apres un
#     nouvel entrainement)
#   - POST /predict {horizon, dju_forecast?} : projette horizon mois
#     au-dela du dernier mois observe ; si dju_forecast est fourni
#     (meme longueur que horizon), reconstruit aussi la consommation
#     reelle prevue
#   - GET /health, GET /metrics
#
# Variables / constantes importantes :
#   - ARTIFACTS_PATH (via services.model_loader)
#   - Compteurs/histos Prometheus : PREDICT_REQUEST_COUNT, PREDICT_LATENCY
#
# Dependances externes : fastapi, pydantic, prometheus_client,
# prometheus_fastapi_instrumentator, numpy, pandas, services.* (interne)
# ============================================================
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from services import forecast, model_loader

app = FastAPI(title="Predict API — elec_prediction", description="Electricity consumption forecasting", version="1.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

PREDICT_REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
PREDICT_LATENCY = Histogram("predict_request_latency_seconds", "Prediction request latency")

# --- Globals loaded at startup ---
_state = {"model": None, "model_type": None, "metadata": None, "ols_model": None}


def _load_artifacts():
    model, model_type, metadata = model_loader.load_deployment_model()
    ols_model = model_loader.load_ols_model()
    _state.update({"model": model, "model_type": model_type, "metadata": metadata, "ols_model": ols_model})


@app.on_event("startup")
def startup_event():
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        # Service can start before train-api has produced its first artifacts —
        # /predict will 503 until /reload-artifacts is called.
        print(f"[predict-api] No artifacts yet at startup: {e}")


@app.post("/reload-artifacts")
def reload_artifacts():
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "reloaded", "model_type": _state["model_type"], "trained_at": _state["metadata"].get("trained_at")}


class PredictRequest(BaseModel):
    horizon: int = 12
    dju_forecast: Optional[List[float]] = None  # optional projected DJU per future month, to reconstruct real Consommation


@app.post("/predict")
def predict(req: PredictRequest):
    PREDICT_REQUEST_COUNT.inc()
    t0 = time.time()

    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="No trained model loaded — run POST /train on train-api, then POST /reload-artifacts here")

    if req.horizon < 1 or req.horizon > 60:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 60 months")

    if req.dju_forecast is not None and len(req.dju_forecast) != req.horizon:
        raise HTTPException(status_code=400, detail="dju_forecast length must equal horizon")

    model_type = _state["model_type"]
    metadata = _state["metadata"]

    if model_type == "holt_winters":
        pred = forecast.forecast_holt_winters(_state["model"], req.horizon)
        lower = upper = None
    else:
        pred, lower, upper = forecast.forecast_sarima(
            _state["model"], req.horizon, log_transform=metadata.get("sarima_log_transform", True),
        )

    last_month = pd.Timestamp(metadata["last_observed_month"])
    future_months = pd.date_range(last_month, periods=req.horizon + 1, freq="M")[1:]

    consommation_pred = None
    if req.dju_forecast is not None:
        consommation_pred = forecast.reconstruct_consumption(pred, req.dju_forecast, _state["ols_model"])

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

    PREDICT_LATENCY.observe(time.time() - t0)

    return {
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
        "model_loaded": _state["model"] is not None,
        "model_type": _state["model_type"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5011)
