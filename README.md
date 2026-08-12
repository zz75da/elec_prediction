# elec_prediction — Electricity Demand Forecasting (MLOps)

[![CI — Tests & DVC Sync](https://github.com/<your-username>/elec_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/elec_prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [zz75da](https://github.com/zz75da) · z.zeghoud@yahoo.com

MLOps platform for forecasting France's monthly national electricity consumption, corrected for
temperature effects (Degrés Jours Unifiés — DJU) and forecast with Holt-Winters and SARIMA.
Rebuilt from the original `P9_01_notebook.ipynb` (OpenClassrooms P9) into a small, containerized
FastAPI + DVC + MLflow + Prometheus/Grafana stack, reusing the architectural patterns of
[`rakuten_mlops_services`](https://github.com/zz75da/rakuten_z) scaled down to a single lightweight
time-series model instead of a 4-encoder multimodal classifier — the whole repo (code + data +
models) is a few MB, well under the 20 GB target.

**Best model:** SARIMA(0,1,1)(1,1,1)₁₂ on the temperature-corrected, log-transformed series —
**MAPE = 2.22%** on the 2019 out-of-sample backtest (quality-gate threshold: < 10%).

---

## Table of Contents

- [Model Overview](#model-overview)
- [Architecture](#architecture)
- [Services](#services)
- [Quick Start](#quick-start)
- [Service Endpoints](#service-endpoints)
- [Monitoring](#monitoring)
- [Data & Experiment Tracking](#data--experiment-tracking)
- [Test Suite](#test-suite)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)
- [What Was Scaled Down From rakuten_mlops_services, and Why](#what-was-scaled-down-from-rakuten_mlops_services-and-why)

---

## Model Overview

Two-stage pipeline, faithfully ported from the notebook:

1. **Temperature correction** — OLS regression `Consommation ~ const + DJU` (R²≈0.94). The
   DJU-driven component is subtracted: `Conso_correction = Consommation − coef_DJU × DJU`.
2. **Forecasting** on `Conso_correction`, two candidate models trained and backtested on a
   held-out year (2019), best one selected automatically by MAPE:
   - **Holt-Winters** — triple exponential smoothing, `trend='add'`, `seasonal='add'`, period 12.
   - **SARIMA(0,1,1)(1,1,1)₁₂** — fit on `log(Conso_correction)`, forecast exponentiated back.
     Selected via ACF/PACF identification + Ljung-Box whiteness test + Shapiro normality test
     (see `notebooks/P9_01_notebook.ipynb` for the full diagnostic walkthrough).

Both models are refit on the *full* series after backtesting and saved — `predict-api` serves
whichever one won the backtest.

---

## Architecture

```
┌─────────────┐
│  Streamlit  │  :8501  — train trigger, forecast charts, historical data
└──────┬──────┘
       │ HTTP
       ▼
┌────────────────┐  POST /train        ┌─────────────────────────────────────────┐
│                 │────────────────────►│  train-api  :5010                       │
│  (user / CI /   │◄── poll status      │  data_loader → preprocess (OLS+DJU) →   │
│   DVC pipeline) │                     │  Holt-Winters + SARIMA → backtest 2019  │
│                 │                     │  → select best → refit on full series   │
└─────────────────┘                     │  → save artifacts → MLflow (DagsHub)    │
                                         └───────────────┬──────────────────────────┘
                                                          │ shared volume: data/artifacts
                                                          ▼
                                         ┌──────────────────────────────────────────┐
                                         │  predict-api  :5011                       │
                                         │  loads best model + OLS model             │
                                         │  POST /predict {horizon, dju_forecast?}   │
                                         └───────────────┬──────────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Prometheus (:9090) ◄── scrapes train-api + predict-api /metrics            │
│  Grafana    (:3000) ──► MAPE/RMSE per model, OLS R², request latency        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Role |
|---|---|---|
| **train-api** | 5010 | Preprocessing + Holt-Winters/SARIMA training, backtest, quality gate |
| **predict-api** | 5011 | Serves forecasts from the best-selected model |
| **streamlit** | 8501 | UI: trigger training, view forecasts, browse historical data |
| **prometheus** | 9090 | Metrics scraping (15s interval) |
| **grafana** | 3000 | Dashboards: MAPE/RMSE by model, service health, latency |
| **MLflow** | — | DagsHub-hosted experiment tracking (same pattern as rakuten_mlops_services) |

No JWT gateway, Airflow, or Kubernetes here — see [why](#what-was-scaled-down-from-rakuten_mlops_services-and-why).

---

## Quick Start

### 1 — Clone and configure

```bash
git clone https://github.com/<your-username>/elec_prediction.git
cd elec_prediction
cp .env.template .env       # fill in DAGSHUB_USER, DAGSHUB_TOKEN (optional — MLflow logging is best-effort)
```

### 2 — Start the stack

```bash
docker compose build
docker compose up -d
```

### 3 — Train, then forecast

```bash
curl -X POST http://localhost:5010/train
# poll: curl http://localhost:5010/train/status/<job_id>

curl -X POST http://localhost:5011/reload-artifacts
curl -X POST http://localhost:5011/predict -H "Content-Type: application/json" -d '{"horizon": 12}'
```

Or open **http://localhost:8501** (Streamlit) and use the "Entraînement" / "Prévision" pages.

### 4 — Verify

```bash
docker compose ps
curl http://localhost:5010/health
curl http://localhost:5011/health
```

Grafana: **http://localhost:3000** (admin / value of `GF_SECURITY_ADMIN_PASSWORD`, default `admin`).

---

## Service Endpoints

### train-api

```
POST /train                    → 202 {"job_id": "...", "status": "running"} | 409 if a job is already running
GET  /train/status/{job_id}    → {"status": "running|success|failed", "best_model": "...", "metrics": {...}}
POST /quality-gate              Runs train-api/tests/test_model_quality.py — 422 if the model regressed
GET  /health
GET  /metrics                   Prometheus: model_mape, model_rmse, model_best_selected, ols_temperature_r2
```

### predict-api

```
POST /predict   {"horizon": 12, "dju_forecast": [optional, len==horizon]}
                → {"model_used": "sarima", "forecast": [{"month": "2020-01", "conso_correction_pred_gwh": ..., "ci_lower_gwh": ..., "ci_upper_gwh": ...}, ...]}
                  dju_forecast, if provided, adds "consommation_pred_gwh" (reconstructed real forecast)
POST /reload-artifacts   Reload the model from disk after a new /train run
GET  /health
GET  /metrics            Prometheus: predict_requests_total, predict_request_latency_seconds
```

---

## Monitoring

Grafana dashboard (`monitoring/grafana/dashboards/elec_dashboard.json`, auto-provisioned) shows:
MAPE/RMSE per model, which model is currently deployed, OLS R², predict-api P95 latency and
request rate, and service up/down. Alert rules (`monitoring/alert-rules.yml`): service down ≥ 3
min, backtest MAPE > 10% for either model, P95 latency > 2s.

---

## Data & Experiment Tracking

| Resource | Source |
|---|---|
| `data/raw/consommation_mensuelle.csv` | [RTE Open Data](https://opendata.reseaux-energies.fr/explore/dataset/equilibre-national-mensuel-prod-conso-brute/) — national monthly consumption (GWh), committed to git (132 rows, a few KB) |
| `data/raw/dju_mensuel.csv` | Degrés Jours Unifiés, averaged across 8 French regions — committed to git |
| `data/processed/`, `data/artifacts/` | DVC-tracked pipeline outputs (preprocessed series, fitted models, metrics) |
| MLflow | DagsHub-hosted, same pattern as rakuten_mlops_services — set `MLFLOW_TRACKING_URI` in `.env` |

Raw data is committed directly (it's tiny); DVC tracks the *pipeline outputs* — this is the
opposite split from rakuten_mlops_services, where the 85k-row multimodal dataset had to be
DVC-tracked from the start. See `dvc.yaml` for the two-stage pipeline (`preprocess`,
`train_and_evaluate`) and `scripts/extract_raw_data.py` to regenerate the raw CSVs from the
original source files if you have them.

---

## Test Suite

```bash
# Unit tests — pure functions, synthetic data, no Docker/DVC needed
pytest tests/unit/ -v

# Integration tests — real FastAPI apps via TestClient, real data/raw/*.csv
pytest tests/integration/ -v -m integration

# Model quality gate — asserts backtest MAPE < 10% (params.yaml: quality_gate.mape_threshold)
python scripts/run_quality_gate_ci.py && pytest train-api/tests/test_model_quality.py -v
```

| File | Scope |
|---|---|
| `tests/unit/test_preprocess.py` | OLS temperature regression + seasonal decomposition |
| `tests/unit/test_trainers.py` | Holt-Winters / SARIMA fit + forecast shapes and confidence intervals |
| `tests/unit/test_evaluate.py` | RMSE / MAPE formulas against known values |
| `tests/unit/test_artifacts.py` | Save/load roundtrip, SHA256 sidecars, best-model selection |
| `tests/integration/test_api_integration.py` | Health checks + full train→predict cycle via TestClient |
| `train-api/tests/test_model_quality.py` | Quality gate: MAPE floor, no missing metrics |

---

## Repository Structure

```
elec_prediction/
├── train-api/
│   ├── app.py                      # /train · /train/status/{id} · /quality-gate · /health · /metrics
│   ├── services/
│   │   ├── data_loader.py          # load consommation_mensuelle.csv + dju_mensuel.csv
│   │   ├── preprocess.py           # OLS Consommation~DJU + seasonal_decompose (notebook cells 25-47)
│   │   ├── trainer_holtwinters.py  # ExponentialSmoothing (notebook cell 56)
│   │   ├── trainer_sarima.py       # SARIMA(0,1,1)(1,1,1)12, log-transform (notebook cells 90-96)
│   │   ├── evaluate.py             # RMSE, MAPE (notebook cells 95-96)
│   │   └── artifacts.py            # atomic save/load + SHA256
│   └── tests/test_model_quality.py
├── predict-api/
│   ├── app.py                      # /predict · /reload-artifacts · /health · /metrics
│   └── services/{model_loader,forecast}.py
├── streamlit/app_streamlit.py      # UI: train trigger, forecast charts, historical data
├── monitoring/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   └── grafana/                    # provisioning + dashboard JSON
├── data/
│   ├── raw/                        # committed (small): consommation_mensuelle.csv, dju_mensuel.csv
│   ├── processed/                  # DVC-tracked: conso_corrigee.csv
│   └── artifacts/                  # DVC-tracked: *.pkl models, model_metadata.json, train_history.json
├── scripts/
│   ├── extract_raw_data.py         # regenerate data/raw from the original xlsx/csv source files
│   └── run_quality_gate_ci.py      # fit + quality gate without booting the FastAPI server (used in CI)
├── notebooks/P9_01_notebook.ipynb  # original EDA/modeling notebook, kept for reference
├── tests/                          # unit + integration suites
├── .github/workflows/ci.yml        # tests + DVC remote sync check
├── dvc.yaml · params.yaml          # pipeline definition + tunable parameters
└── docker-compose.yml              # train-api, predict-api, streamlit, prometheus, grafana
```

---

## Environment Variables

Copy `.env.template` to `.env`:

| Variable | Description |
|---|---|
| `DAGSHUB_USER` / `DAGSHUB_TOKEN` | DagsHub credentials (DVC remote + MLflow) |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/elec_prediction.mlflow` |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Same as `DAGSHUB_USER`/`DAGSHUB_TOKEN` |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password |

MLflow logging in `train-api/app.py` is **best-effort**: if `MLFLOW_TRACKING_URI` isn't set, training
still runs and artifacts still save — only the MLflow run logging is skipped.

---

## What Was Scaled Down From rakuten_mlops_services, and Why

`rakuten_mlops_services` serves a 4-encoder multimodal (text+image) classifier over ~85k products
and needs Airflow orchestration, a JWT gate-api, dedicated GPU-ish encoder containers, and
Prometheus/Grafana/Alertmanager at full scale (14 services, ~20 GB with data+models).
`elec_prediction` forecasts a single 132-row monthly time series with two lightweight statistical
models (no GPU, no large embeddings) — so this repo keeps the parts of that architecture that
still pay for themselves at this scale:

- **kept:** FastAPI train/predict split, async job registry pattern, DVC pipeline, MLflow/DagsHub
  tracking, Docker Compose, Prometheus + Grafana + alert rules, pytest unit/integration/quality-gate
  structure, GitHub Actions CI.
- **dropped:** JWT gate-api (no multi-user auth need for a single-model demo), Airflow (a `/train`
  call or a cron job is enough for a model that retrains in seconds, not hours), Kubernetes PoC
  (nothing here needs horizontal scaling), Alertmanager/pushgateway/minio/postgres (no batch jobs,
  no object storage needed for KB-sized artifacts).

---

## Author & License

**Author:** [Zobir Zeghoud](https://github.com/zz75da) — z.zeghoud@yahoo.com

Licensed under the [MIT License](LICENSE).
