# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : projection (forecast) a partir d'un modele Holt-Winters
# ou SARIMA deja entraine, avec reconstruction optionnelle de la
# consommation reelle a partir de la consommation corrigee de
# l'effet temperature (inverse de train-api/services/preprocess.py
# fit_temperature_regression).
#
# Fonctions principales :
#   - forecast_holt_winters(model, horizon) -> np.ndarray
#   - forecast_sarima(results, horizon, log_transform, alpha) ->
#     (pred, ci_lower, ci_upper)
#   - reconstruct_consumption(conso_correction_pred, dju_forecast, ols_model)
#     -> Consommation = Conso_correction + coef_DJU * DJU
#     (inverse de Conso_correction = Consommation - coef_DJU * DJU)
#
# Dependances externes : numpy
# ============================================================
import numpy as np


def forecast_holt_winters(model, horizon: int) -> np.ndarray:
    return np.asarray(model.forecast(horizon))


def forecast_sarima(results, horizon: int, log_transform: bool = True, alpha: float = 0.05):
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


def reconstruct_consumption(conso_correction_pred: np.ndarray, dju_forecast, ols_model) -> np.ndarray:
    """Consommation = Conso_correction + coef_DJU * DJU (inverse of the training-time correction)."""
    coef_dju = ols_model.params["DJU"]
    dju = np.asarray(dju_forecast)
    return np.asarray(conso_correction_pred) + coef_dju * dju
