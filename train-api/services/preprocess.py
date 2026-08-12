# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : reproduit fidelement le pipeline de correction
# temperature + desaisonnalisation de P9_01_notebook.ipynb
# (cellules 25-47).
#
# Fonctions principales :
#   - fit_temperature_regression(df) -> (ols_results, df_avec_correction) :
#     OLS  Consommation ~ const + DJU  (cellule 25), puis
#     Conso_correction = Consommation - coef_DJU * DJU  (cellule 38)
#   - deseasonalize(df) -> df avec colonnes seasonal/trend/CVS :
#     seasonal_decompose additif sur Conso_correction (cellules 41-47)
#   - run_preprocessing(df_merged) -> (df_complet, stats) : orchestre
#     les deux etapes et retourne un dict de stats (R2, p-values, ...)
#     pour le tracking MLflow / logs
#
# Variables / constantes importantes :
#   - aucune (les DataFrame en entree/sortie portent l'index Mois)
#
# Dependances externes : numpy, pandas, statsmodels
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


def fit_temperature_regression(df: pd.DataFrame):
    """
    Notebook cells 25-38:
        y = Consommation ; x = const + DJU
        linReg = OLS(y, x).fit()
        Conso_correction = Consommation - coef_DJU * DJU
    """
    y = df["Consommation"]
    x = sm.add_constant(df["DJU"])
    ols_results = sm.OLS(y, x).fit()

    df = df.copy()
    df["Conso_correction"] = df["Consommation"] - df["DJU"] * ols_results.params["DJU"]
    return ols_results, df


def deseasonalize(df: pd.DataFrame, seasonal_periods: int = 12) -> pd.DataFrame:
    """
    Notebook cells 41-47: additive seasonal decomposition of Conso_correction.
    Adds 'seasonal', 'trend' and 'CVS' (deseasonalized) columns.
    """
    df = df.copy()
    series = df["Conso_correction"].dropna()
    decomp = seasonal_decompose(series, model="additive", period=seasonal_periods)
    df["seasonal"] = decomp.seasonal
    df["trend"] = decomp.trend
    df["CVS"] = series - decomp.seasonal
    return df


def run_preprocessing(df_merged: pd.DataFrame, seasonal_periods: int = 12):
    """
    Full preprocessing pipeline (regression + deseasonalization).
    Returns (df_processed, stats) where stats is a flat dict suitable for
    MLflow log_params/log_metrics and the train-api job record.
    """
    ols_results, df = fit_temperature_regression(df_merged)
    df = deseasonalize(df, seasonal_periods=seasonal_periods)

    stats = {
        "n_observations": int(len(df)),
        "date_min": str(df.index.min().date()),
        "date_max": str(df.index.max().date()),
        "ols_r2": round(float(ols_results.rsquared), 4),
        "ols_coef_dju": round(float(ols_results.params["DJU"]), 4),
        "ols_coef_const": round(float(ols_results.params["const"]), 4),
        "ols_pvalue_dju": round(float(ols_results.pvalues["DJU"]), 6),
        "ols_shapiro_pvalue": round(float(_shapiro_pvalue(ols_results.resid)), 6),
    }
    return df, ols_results, stats


def _shapiro_pvalue(resid) -> float:
    from scipy.stats import shapiro
    _, p = shapiro(resid)
    return p
