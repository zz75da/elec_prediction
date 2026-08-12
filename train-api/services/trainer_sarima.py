# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : modele SARIMA(0,1,1)(1,1,1)12, reproduit de
# P9_01_notebook.ipynb cellules 77-96 (identification + a-posteriori
# analysis). C'est le modele retenu en production (MAPE=2.22% sur
# l'annee 2019 tronquee).
#
# Fonctions principales :
#   - fit_sarima(train_series, order, seasonal_order, log_transform)
#     -> (SARIMAXResults, log_transform utilise) : fit sur
#     log(serie) si log_transform=True (cellule 90-92, c'est la
#     version qui obtient MAPE=2.22%), sinon sur la serie brute
#     (cellule 77-81, identification des parametres)
#   - forecast(results, horizon, log_transform) -> (pred, ci_lower, ci_upper) :
#     get_forecast(horizon), exponentiation si log_transform=True
#     (cellule 93)
#
# Dependances externes : numpy, statsmodels
# ============================================================
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(train_series, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12), log_transform: bool = True):
    """
    Notebook cells 90-92 (a-posteriori / production fit):
        y_tronc = np.log(x_tronc)
        model2tronc = SARIMAX(y_tronc, order=(0,1,1), seasonal_order=(1,1,1,12)).fit()
    log_transform=True reproduces the model that scored MAPE=2.22% on the 2019 holdout.
    """
    series = np.log(np.asarray(train_series)) if log_transform else np.asarray(train_series)
    model = SARIMAX(series, order=tuple(order), seasonal_order=tuple(seasonal_order))
    results = model.fit(disp=False)
    return results


def forecast(results, horizon: int, log_transform: bool = True, alpha: float = 0.05):
    """Notebook cell 93: get_forecast(12), exponentiate mean + CI back if log_transform."""
    pred_obj = results.get_forecast(horizon)
    mean = pred_obj.predicted_mean
    conf_int = pred_obj.conf_int(alpha=alpha)
    lower = np.asarray([row[0] for row in conf_int])
    upper = np.asarray([row[1] for row in conf_int])

    if log_transform:
        mean = np.exp(mean)
        lower = np.exp(lower)
        upper = np.exp(upper)

    return np.asarray(mean), lower, upper
