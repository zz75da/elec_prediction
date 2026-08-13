# elec_prediction — Electricity Demand Forecasting (MLOps)

[![CI — Tests & DVC Sync](https://github.com/zz75da/elec_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/zz75da/elec_prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [zz75da](https://github.com/zz75da) · z.zeghoud@yahoo.com

MLOps platform for forecasting monthly national electricity consumption, corrected for
temperature effects (heating degree-days) and forecast with Holt-Winters, SARIMA, and a pooled
multi-country LightGBM model. Started as a France-only forecasting project and now supports
**7 countries** (France, USA, Germany, Austria, Luxembourg, UK, Finland — selectable from the
Streamlit UI), each trained and served independently. Small, containerized FastAPI + DVC + MLflow
+ Prometheus/Grafana stack — the whole repo (code + data + models) is a few MB.

**Best model (France):** the pooled global LightGBM model (`ml_global`) on the
temperature-corrected series — **MAPE = 1.93%** on the 2019 out-of-sample backtest
(SARIMA: 1.97%, Holt-Winters: 2.65% — quality-gate threshold: < 10%).

---

## Table of Contents

- [Why This Stack](#why-this-stack)
- [Model Overview](#model-overview)
- [Architecture](#architecture)
- [Services](#services)
- [Quick Start](#quick-start)
- [Service Endpoints](#service-endpoints)
- [Monitoring](#monitoring)
- [Multi-Country Data](#multi-country-data)
- [Data & Experiment Tracking](#data--experiment-tracking)
- [Test Suite](#test-suite)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)

---

## Why This Stack

Every non-obvious infrastructure choice below has a reason and, where it costs something, an
explicit note on how that cost is handled — not just a list of technologies.

| Choice | Why | Trade-off & how it's handled |
|---|---|---|
| **FastAPI, split into `train-api` + `predict-api`** | Async-capable, free request validation + OpenAPI docs via Pydantic. Splitting training (slow, resource-heavy statsmodels/LightGBM fits) from serving (latency-sensitive) means a training run never blocks or competes with prediction requests. | Two services have to be kept in sync — `predict-api` doesn't automatically see a freshly trained model. Handled with an explicit `POST /reload-artifacts` call, which Streamlit fires automatically right after a training run succeeds, rather than a shared database or polling. |
| **In-memory threaded job registry, not Celery/RQ** | Training one country takes seconds; a full task queue (broker + worker pool) is disproportionate infrastructure for that workload. | An in-memory registry doesn't survive a container restart. Handled by also persisting every job's result to `data/jobs/*.json` and recovering it on startup — an interrupted job is marked `"interrupted"` explicitly rather than silently lost. |
| **DVC for pipeline/artifact versioning** | Reproducible pipeline (`dvc repro`), visualized as a DAG on DagsHub, without bloating git history with binary model files. | Can drift out of sync with reality if a pipeline's scope creeps. Handled by keeping it narrow and honest: `dvc.yaml` covers France's pipeline only (documented as such), and raw data is committed directly rather than DVC-tracked, since it's a few hundred KB. |
| **MLflow tracking via DagsHub, not self-hosted** | Real experiment tracking (params/metrics/tags per run) without running an MLflow server + Postgres + object storage. | The full `mlflow` package hard-depends on `pyarrow` — which turned out to crash LightGBM natively on Windows the moment both were imported in the same process (a real incident hit and isolated while building this, not a hypothetical). Handled by switching to `mlflow-skinny`, same tracking-client API, none of the unused server/UI dependencies. Logging is also best-effort: training never fails just because DagsHub is unreachable. |
| **Prometheus + Grafana** | The same observability pattern a production stack would use — per-country, per-model MAPE/RMSE, service health, and request latency, queryable and dashboarded rather than only logged. | Two extra containers for what's still a small project. Accepted deliberately: they're free and tiny, and the point is to demonstrate the real pattern rather than skip observability because current scale doesn't strictly demand it. |
| **Docker Compose, not Kubernetes** | Five services on one machine; nothing here needs horizontal scaling, rolling updates, or a service mesh. | A straightforward scope match rather than a trade-off — revisit only if that stops being true. |
| **Streamlit for the UI** | A working multi-page interface (training trigger, forecasting, historical browsing, cross-country analytics) with no separate frontend build step. | Streamlit re-runs the whole script on every interaction, which would mean re-fetching or recomputing on every click. Handled with `@st.cache_data(ttl=...)` on the country list and the 2019 comparison analysis, so repeat interactions don't re-hit the APIs or reprocess CSVs each time. |
| **Statistical models + one pooled global LightGBM, not deep learning** | With ~70-130 monthly rows per country, classical models and a shallow pooled gradient-boosted model are the evidence-backed choice for this data shape (M5-competition findings, the Global Forecasting Models literature) — deep learning needs far more series/data to avoid overfitting here. | Not state-of-the-art next to 2024-2025 foundation time-series models. Handled by researching and documenting that trade-off explicitly rather than ignoring it (see [Model Overview](#model-overview)) — N-BEATS/N-HiTS and Chronos/TimeGPT were evaluated and consciously deferred, not overlooked. |
| **Data connectors kept out of the running containers** | `train-api`/`predict-api` never need third-party API credentials or outbound network access — only local files (see [Multi-Country Data](#multi-country-data)). | Syncing a country's data becomes a separate, manual-or-scheduled step rather than automatic. Handled with one documented entry point (`scripts/sync_country_data.py`) that can be put on a schedule (cron/GitHub Actions) if automated refresh is wanted later. |
| **`predict-api` duplicates `train-api`'s model-loading code** rather than sharing a library | Genuine service independence — `predict-api` can be deployed, restarted, or scaled without depending on `train-api`'s codebase at all. | A change to the artifact format has to be applied in two places. Accepted as a deliberate trade-off favoring independence over DRY. |

---

## Model Overview

Two-stage pipeline:

1. **Temperature correction** — OLS regression `Consommation ~ const + DJU` (R²≈0.94). The
   DJU-driven component is subtracted: `Conso_correction = Consommation − coef_DJU × DJU`.
2. **Forecasting** on `Conso_correction`, three candidate models trained and backtested on a
   held-out year, best one selected automatically by MAPE:
   - **Holt-Winters** — triple exponential smoothing, `trend='add'`, `seasonal='add'`, period 12.
   - **SARIMA(0,1,1)(1,1,1)₁₂** — fit on `log(Conso_correction)`, forecast exponentiated back.
     Selected via ACF/PACF identification + Ljung-Box whiteness test + Shapiro normality test.
   - **`ml_global`** — a single LightGBM model (via Nixtla's `mlforecast`) trained on **all 7
     countries' series pooled together**, not one ML model per country. With only ~70-130 monthly
     rows per country, a per-country ML model would overfit badly; pooling is the well-evidenced
     fix for exactly this data shape (M5-competition findings, the "Global Forecasting Models"
     literature — shallow/regularized global models tolerate few series far better than deep
     learning does). See `train-api/services/trainer_ml.py`'s module docstring for the backtest
     methodology (each country's own `test_year` cutoff is respected — no country's held-out rows
     ever leak into another's backtest fit). Skips cleanly (`model_ml_global.min_countries_required`
     in `params.yaml`) rather than crashing if too few countries have synced data yet.

All three candidates are refit on the *full* series (the ML candidate on the full pooled dataset)
after backtesting and saved — `predict-api` serves whichever one won the backtest, per country.

**Considered, deferred**: deep learning (N-BEATS/N-HiTS) and 2024-2025 foundation time-series
models (Chronos, TimeGPT) were researched but not implemented — both need far more series/data
than 7 countries provide to pay off, and Chronos in particular would pull in PyTorch, breaking
this repo's lightweight-Docker-image ethos for an unclear accuracy win at this scale.

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
| **MLflow** | — | DagsHub-hosted experiment tracking |

No JWT gateway, Airflow, or Kubernetes here — the training workload is small enough (seconds per
run) that a `/train` call or a cron job covers it without that infrastructure.

---

## Quick Start

### 1 — Clone and configure

```bash
git clone https://github.com/zz75da/elec_prediction.git
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
POST /train   {"country": "france"}   optional, defaults to params.yaml's default_country
                → 202 {"job_id": "...", "status": "running", "country": "france"}
                  400 if country is unknown | 409 if a job is already running
GET  /train/status/{job_id}    → {"status": "running|success|failed", "country": ..., "best_model": "...", "metrics": {...}}
POST /quality-gate?country=france   Runs train-api/tests/test_model_quality.py — 422 if the model regressed
GET  /countries                 → {"countries": [{"code": "france", "label": "France"}, ...], "default_country": "france"}
GET  /health
GET  /metrics                   Prometheus: model_mape, model_rmse, model_best_selected, ols_temperature_r2
                                 (all labeled by model AND country)
```

### predict-api

```
POST /predict   {"horizon": 12, "dju_forecast": [optional, len==horizon], "country": "france"}
                → {"country": "france", "model_used": "sarima", "forecast": [{"month": "2020-01", "conso_correction_pred_gwh": ..., "ci_lower_gwh": ..., "ci_upper_gwh": ...}, ...]}
                  dju_forecast, if provided, adds "consommation_pred_gwh" (reconstructed real forecast)
                  503 if that country hasn't been trained yet
POST /reload-artifacts?country=france   Reload one country's model from disk after a new /train run
                                          (omit country to reload every country found on disk)
GET  /health                    → includes model_loaded_countries: [...]
GET  /metrics                   Prometheus: predict_requests_total, predict_request_latency_seconds (labeled by country)
```

All country params default to `"france"` if omitted — every endpoint above works exactly as
before for callers that don't pass one.

---

## Monitoring

Grafana dashboard (`monitoring/grafana/dashboards/elec_dashboard.json`, auto-provisioned) shows:
MAPE/RMSE per model, which model is currently deployed, OLS R², predict-api P95 latency and
request rate, and service up/down. Alert rules (`monitoring/alert-rules.yml`): service down ≥ 3
min, backtest MAPE > 10% for any candidate model, training stale > 30 days, P95 latency > 2s.

---

## Multi-Country Data

Every layer (raw data, artifacts, Prometheus labels, the Streamlit UI) is namespaced by a
`country` code so each country trains and serves an independent model, sharing only the
methodology (OLS temperature correction + Holt-Winters/SARIMA backtest-and-select).

| Country | Source | Auth | Status |
|---|---|---|---|
| France | [RTE Open Data](https://opendata.reseaux-energies.fr/explore/dataset/equilibre-national-mensuel-prod-conso-brute/), static historical export | none | committed (132 months, 2009–2019) |
| Germany | [SMARD](https://www.smard.de) (Bundesnetzagentur grid load, region=DE, filter 410) | none | committed (72 months, 2019–2024) |
| Austria | [SMARD](https://www.smard.de) (region=AT — same platform as Germany) | none | committed (120 months, 2015–2024) |
| Luxembourg | [SMARD](https://www.smard.de) (region=LU — same platform as Germany) | none | committed (120 months, 2015–2024) |
| USA | [EIA API v2](https://www.eia.gov/opendata/) retail-sales, natively monthly | free instant key | committed (120 months, 2015–2024) |
| UK | [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data) (national demand, half-hourly) | none | committed (120 months, 2015–2024) |
| Finland | [Fingrid](https://data.fingrid.fi) dataset 124, consumption (15-min) | free instant key | committed (120 months, 2015–2024) |

All 7 countries are synced and trained — best-candidate backtest MAPE ranges from 1.3%
(Germany) to 8.9% (Luxembourg, a much smaller/noisier grid), all under the 10% quality-gate
threshold. Re-run `scripts/sync_country_data.py` any time to refresh a country's data with a
later end date.

Germany, Austria, and Luxembourg all share the same connector (`connectors/smard.py`) — SMARD is
one platform serving all three (a legacy of the joint DE-AT-LU bidding zone before Austria split
out in 2018), parameterized by SMARD's `region` code rather than three near-duplicate files.

Temperature/degree-days for the 6 non-France countries come from the free, no-key
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
converted to heating-degree-days with the same convention as the French DJU series
(`connectors/degree_days.py`) — so `preprocess.py`'s OLS math never changes, only where the
DJU numbers come from.

**Architecture**: `connectors/` (top-level package, adapter pattern — one `CountryConnector`
subclass per country/platform, picked via `connectors/registry.py`) is **host-run only**, never
imported by the `train-api`/`predict-api` containers — neither service needs third-party API
credentials or outbound network access, only local files. `scripts/sync_country_data.py --country
<code>` fetches + validates (via `train-api/services/schemas.py`'s pydantic checks) + writes
`data/raw/<country>/*.csv`, which `train-api` then reads exactly like France's static CSVs.

```bash
cp .env.template .env             # fill in EIA_API_KEY / FINGRID_API_KEY (SMARD-based countries need no key)
python scripts/sync_country_data.py --country usa --start 2015-01 --end 2024-12
# or sync every wired-up country at once:
python scripts/sync_country_data.py --country all --start 2015-01 --end 2024-12
```

After syncing, adjust that country's `test_year` in `params.yaml` to a year with enough
post-test-year history for a meaningful backtest, then `POST /train {"country": "usa"}`.
`tests/data_quality/` validates whichever countries are currently synced and skips the rest —
CI stays green regardless of which API keys have been supplied.

**"2019 Comparison" page** — 2019 is the only calendar year every country has a complete
series for (France's static export ends there; the other six's synced range starts there),
so it's the one honest apples-to-apples comparison point. The page reads all 7 countries'
raw CSVs directly (no training required) and computes, live: total, per-capita, and
GDP-normalized ("energy intensity") 2019 consumption (World Bank population/GDP estimates —
per-capita and intensity answer different questions, a lifestyle-volume measure vs. an
economic-efficiency one, and reporting only one is a recognized gap in this kind of
comparison); the monthly seasonality profile as a % of each country's own annual total;
a Pearson correlation heatmap of those seasonality profiles (which countries share the same
month-to-month shape, independent of their consumption levels); and each country's
temperature sensitivity — the slope and R² of `Consommation ~ DJU`, with proper OLS
inference (standard error, 95% CI, p-value on the slope) since n=12 per country is a real
statistical-power limit, shown as small-multiple scatter panels plus an explicit
Methodology & Limitations note. The one finding this consistently surfaces: the USA is the
only country where summer (Jun–Aug) outweighs winter (Dec–Feb) in the annual total — and its
DJU slope's 95% CI actually crosses zero (p≈0.19), confirming this repo's heating-only
degree-day model has no term for the air-conditioning-driven demand that explains it,
rather than USA demand being genuinely temperature-insensitive.

---

## Data & Experiment Tracking

| Resource | Source |
|---|---|
| `data/raw/<country>/consommation_mensuelle.csv` | Per-country source, see [Multi-Country Data](#multi-country-data) — France committed directly (132 rows, a few KB); others via `scripts/sync_country_data.py` |
| `data/raw/<country>/dju_mensuel.csv` | France: Degrés Jours Unifiés, averaged across 8 regions. Others: Open-Meteo-derived heating-degree-days |
| `data/processed/<country>/`, `data/artifacts/<country>/` | DVC-tracked pipeline outputs (preprocessed series, fitted models, metrics), namespaced per country |
| MLflow | DagsHub-hosted — set `MLFLOW_TRACKING_URI` in `.env`, runs tagged with `country` |

Raw data is committed directly (it's tiny); DVC tracks the *pipeline outputs* instead. See
`dvc.yaml` for the two-stage pipeline (`preprocess`, `train_and_evaluate`, scoped to France) and
`scripts/extract_raw_data.py` to regenerate France's raw CSVs from the original source files if
you have them.

---

## Test Suite

```bash
# Data-quality gate — validates each configured country's raw CSVs (offline, no network);
# skips countries not yet synced instead of failing
pytest tests/data_quality/ -v

# Unit tests — pure functions, synthetic data, no Docker/DVC needed
pytest tests/unit/ -v

# Integration tests — real FastAPI apps via TestClient, real data/raw/*.csv
pytest tests/integration/ -v -m integration

# Model quality gate — asserts backtest MAPE < 10% (params.yaml: quality_gate.mape_threshold)
python scripts/run_quality_gate_ci.py --country france && pytest train-api/tests/test_model_quality.py -v
```

| File | Scope |
|---|---|
| `tests/data_quality/test_raw_data_quality.py` | Per-country schema validation + test_year sanity, skips unsynced countries |
| `tests/unit/test_preprocess.py` | OLS temperature regression + seasonal decomposition |
| `tests/unit/test_trainers.py` | Holt-Winters / SARIMA fit + forecast shapes and confidence intervals |
| `tests/unit/test_trainer_ml.py` | Pooled multi-country LightGBM: pooling/truncation, fit+forecast round-trip, graceful skip on a bad country |
| `tests/unit/test_evaluate.py` | RMSE / MAPE formulas against known values |
| `tests/unit/test_artifacts.py` | Save/load roundtrip, SHA256 sidecars, best-model selection, multi-country namespace isolation |
| `tests/unit/test_schemas.py` | Pydantic ingestion-boundary validation (schemas.py) |
| `tests/unit/test_config.py` | Country config resolution (config.py) |
| `tests/unit/test_connectors_degree_days.py` | Daily temperature → monthly DJU conversion |
| `tests/integration/test_api_integration.py` | Health checks, `/countries`, full train→predict cycle via TestClient |
| `train-api/tests/test_model_quality.py` | Quality gate: MAPE floor, no missing metrics (per `TRAIN_COUNTRY` env var, default france) |

---

## Repository Structure

```
elec_prediction/
├── train-api/
│   ├── app.py                      # /train · /train/status/{id} · /quality-gate · /countries · /health · /metrics
│   ├── services/
│   │   ├── data_loader.py          # load consommation_mensuelle.csv + dju_mensuel.csv, per country
│   │   ├── preprocess.py           # OLS Consommation~DJU + seasonal_decompose (notebook cells 25-47)
│   │   ├── trainer_holtwinters.py  # ExponentialSmoothing (notebook cell 56)
│   │   ├── trainer_sarima.py       # SARIMA(0,1,1)(1,1,1)12, log-transform (notebook cells 90-96)
│   │   ├── trainer_ml.py           # ml_global: pooled multi-country LightGBM via mlforecast
│   │   ├── evaluate.py             # RMSE, MAPE (notebook cells 95-96)
│   │   ├── artifacts.py            # atomic save/load + SHA256, namespaced under ARTIFACTS_PATH/<country>/
│   │   ├── schemas.py              # pydantic ingestion-boundary validation (DataQualityError)
│   │   └── config.py               # country config resolution from params.yaml
│   └── tests/test_model_quality.py
├── predict-api/
│   ├── app.py                      # /predict · /reload-artifacts · /health · /metrics — per-country model state
│   └── services/{model_loader,forecast}.py
├── streamlit/app_streamlit.py      # UI (English): country selector w/ flags, train trigger,
│                                    # forecast charts, historical data, "2019 Comparison" page
├── connectors/                     # host-run only — never imported by train-api/predict-api containers
│   ├── registry.py                 # ConnectorSpec per country (adapter pattern, config-driven)
│   ├── base.py, open_meteo.py, degree_days.py
│   └── usa_eia.py, smard.py (DE/AT/LU), uk_neso.py, finland_fingrid.py
├── monitoring/
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   └── grafana/                    # provisioning + dashboard JSON
├── data/
│   ├── raw/<country>/              # committed (small): consommation_mensuelle.csv, dju_mensuel.csv
│   ├── processed/<country>/        # DVC-tracked: conso_corrigee.csv
│   └── artifacts/<country>/        # DVC-tracked: *.pkl models, model_metadata.json, train_history.json
├── scripts/
│   ├── extract_raw_data.py         # regenerate data/raw/france from the original xlsx/csv source files
│   ├── sync_country_data.py        # fetch + validate + write data/raw/<country>/ via connectors/
│   └── run_quality_gate_ci.py      # fit + quality gate without booting the FastAPI server (used in CI)
├── tests/                          # data_quality + unit + integration suites
├── .github/workflows/ci.yml        # data quality + tests + DVC remote sync check
├── dvc.yaml · params.yaml          # pipeline definition + tunable parameters (data.countries map)
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
| `EIA_API_KEY` / `FINGRID_API_KEY` | Host-only, used by `scripts/sync_country_data.py` — never reach any container (see [Multi-Country Data](#multi-country-data)) |

MLflow logging in `train-api/app.py` is **best-effort**: if `MLFLOW_TRACKING_URI` isn't set, training
still runs and artifacts still save — only the MLflow run logging is skipped.

---

## Author & License

**Author:** [Zobir Zeghoud](https://github.com/zz75da) — z.zeghoud@yahoo.com

Licensed under the [MIT License](LICENSE).
