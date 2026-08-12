# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : configuration pytest partagee. Ajoute train-api/ et
# predict-api/ au sys.path pour que `import services...` fonctionne
# depuis les tests racine (meme pattern que
# rakuten_mlops_services/tests/conftest.py), et fournit des
# fixtures de donnees synthetiques pour les tests unitaires
# rapides (pas de dependance au vrai CSV).
#
# Fixtures principales :
#   - synthetic_monthly_series(n_years=6) -> DataFrame indexe par
#     Mois avec colonnes Consommation, DJU (signal saisonnier +
#     bruit, deterministe via random_state)
#
# Dependances externes : pytest, pandas, numpy
# ============================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))                # for `import connectors` (top-level package)
sys.path.insert(0, str(REPO_ROOT / "train-api"))  # `import services` -> train-api/services in unit tests
# predict-api is intentionally NOT added here: it defines its own `services` package with
# the same name as train-api's, so tests/integration/test_api_integration.py's fixtures add
# it locally (and purge sys.modules first) only when a test actually needs predict-api's app.


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests requiring a running FastAPI app")


@pytest.fixture
def synthetic_monthly_series() -> pd.DataFrame:
    """Deterministic synthetic monthly series: consumption driven by DJU + noise, no real data needed."""
    rng = np.random.default_rng(42)
    n_years = 6
    months = pd.date_range("2013-01-31", periods=n_years * 12, freq="M")
    month_of_year = months.month
    # DJU: high in winter, ~0 in summer
    dju = np.clip(300 * np.cos((month_of_year - 1) / 12 * 2 * np.pi) + 250 + rng.normal(0, 20, len(months)), 0, None)
    consommation = 35000 + 40 * dju + rng.normal(0, 500, len(months))
    return pd.DataFrame({"Mois": months, "Consommation": consommation, "DJU": dju}).set_index("Mois")
