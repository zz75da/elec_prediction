# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : modele de lissage exponentiel triple (Holt-Winters),
# reproduit de P9_01_notebook.ipynb cellule 56.
#
# Fonctions principales :
#   - fit_holt_winters(train_series, seasonal_periods, trend, seasonal)
#     -> HoltWintersResults : ExponentialSmoothing(...).fit()
#   - forecast(model, horizon) -> np.ndarray : model.forecast(horizon)
#
# Dependances externes : numpy, statsmodels
# ============================================================
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing


def fit_holt_winters(train_series, seasonal_periods: int = 12, trend: str = "add", seasonal: str = "add"):
    """Notebook cell 56: ExponentialSmoothing(seasonal_periods=12, trend='add', seasonal='add').fit()"""
    model = ExponentialSmoothing(
        np.asarray(train_series),
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
    ).fit()
    return model


def forecast(model, horizon: int) -> np.ndarray:
    return np.asarray(model.forecast(horizon))
