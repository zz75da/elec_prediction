# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : troisieme candidat de prevision — un modele "global"
# (LightGBM via mlforecast de Nixtla) entraine sur les series
# Conso_correction de TOUS les pays regroupes (pooling), plutot
# qu'un modele ML par pays isole sur ~100 lignes. Justification :
# recherche academique/industrie (competition M5, litterature des
# "Global Forecasting Models") — un modele global peu profond
# tolere un petit nombre de series bien mieux qu'un modele ML
# entraine isolement par pays sur trop peu de lignes. Cf. le
# fichier de plan associe pour le detail du raisonnement.
#
# Opere sur Conso_correction (la meme cible que Holt-Winters et
# SARIMA) — la composante DJU a deja ete retiree par l'OLS dans
# preprocess.py, donc aucun des 3 candidats n'a besoin du DJU en
# feature d'entree.
#
# Fonctions principales :
#   - load_other_country_frames(params, seasonal_periods, exclude)
#     -> {country: df} pour tous les pays sauf `exclude`, chacun
#     charge + pretraite (data_loader + preprocess), les pays dont
#     les fichiers bruts sont absents sont ignores (log warning,
#     pas d'exception) — ce qui permet a min_countries_required de
#     degrader proprement plutot que de planter
#   - build_pooled_frame(country_frames, target_col, cutoff_years)
#     -> DataFrame long format (unique_id, ds, y, country) pret
#     pour mlforecast ; cutoff_years tronque chaque pays a SA
#     propre annee limite (voir le plan pour la justification —
#     evite qu'un backtest "voie" des annees futures d'un autre
#     pays)
#   - fit_global_model(pooled_df, ml_cfg) -> MLForecast entraine
#     (c'est cet objet complet qui est pickle — il porte l'historique
#     de lags de chaque serie, donc predict-api n'a besoin d'aucun
#     etat supplementaire, comme pour Holt-Winters/SARIMA)
#   - forecast_country(fcst, country, horizon) -> np.ndarray
#
# Dependances externes : pandas, numpy, mlforecast, lightgbm
# ============================================================
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast

from . import data_loader, preprocess

logger = logging.getLogger(__name__)


def load_other_country_frames(params: dict, seasonal_periods: int, exclude: str) -> Dict[str, pd.DataFrame]:
    """Load + preprocess every configured country except `exclude`. A country whose raw
    files are missing/invalid is skipped with a logged warning, not fatal."""
    countries_cfg = params.get("data", {}).get("countries", {})
    frames: Dict[str, pd.DataFrame] = {}
    for code, cfg in countries_cfg.items():
        if code == exclude:
            continue
        try:
            df_merged = data_loader.load_merged(cfg["raw_conso_path"], cfg["raw_dju_path"], country=code)
            df, _ols_results, _stats = preprocess.run_preprocessing(df_merged, seasonal_periods=seasonal_periods)
            frames[code] = df
        except Exception as exc:  # noqa: BLE001 — best-effort, log and skip a bad country rather than fail the whole run
            logger.warning(f"[trainer_ml] Skipping country={code} for global model pooling: {exc}")
    return frames


def build_pooled_frame(
    country_frames: Dict[str, pd.DataFrame],
    target_col: str = "Conso_correction",
    cutoff_years: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Long format (unique_id, ds, y, country) across every country in country_frames.
    cutoff_years, if given, truncates each country's rows to year < cutoff_years[country]
    (exclusive) — a country absent from cutoff_years is included in full."""
    rows = []
    for code, df in country_frames.items():
        sub = df
        if cutoff_years and code in cutoff_years:
            sub = sub[sub.index.year < cutoff_years[code]]
        if sub.empty:
            continue
        rows.append(pd.DataFrame({
            "unique_id": code,
            "ds": sub.index,
            "y": sub[target_col].values,
            "country": code,
        }))
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", "y", "country"])
    pooled = pd.concat(rows, ignore_index=True).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pooled["country"] = pooled["country"].astype("category")
    return pooled


def fit_global_model(pooled_df: pd.DataFrame, ml_cfg: dict) -> MLForecast:
    model = LGBMRegressor(**ml_cfg["lightgbm_params"])
    fcst = MLForecast(
        models={"ml_global": model},
        freq=ml_cfg["freq"],
        lags=ml_cfg["lags"],
        date_features=ml_cfg["date_features"],
    )
    fcst.fit(pooled_df, static_features=ml_cfg.get("static_features", ["country"]))
    return fcst


def forecast_country(fcst: MLForecast, country: str, horizon: int) -> np.ndarray:
    preds = fcst.predict(h=horizon, ids=[country])
    preds = preds.sort_values("ds")
    return preds["ml_global"].to_numpy()
