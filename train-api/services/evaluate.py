# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : metriques d'evaluation reproduites de P9_01_notebook.ipynb
# cellules 95-96 (RMSE, MAPE) utilisees pour le backtest 2019 et
# le quality gate.
#
# Fonctions principales :
#   - rmse(actual, predicted) -> float
#   - mape(actual, predicted) -> float (en pourcentage)
#   - evaluate_forecast(actual, predicted) -> dict {rmse, mape}
#
# Dependances externes : numpy
# ============================================================
import numpy as np


def rmse(actual, predicted) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual, predicted) -> float:
    """Notebook cell 96: mape = mean(|1 - pred/actual|) * 100"""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean(np.abs(1 - predicted / actual)) * 100)


def evaluate_forecast(actual, predicted) -> dict:
    return {
        "rmse": round(rmse(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 4),
    }
