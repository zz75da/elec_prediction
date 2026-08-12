# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : chargement des series mensuelles brutes (consommation
# electrique nationale + DJU) produites par scripts/extract_raw_data.py
# et deja committees dans data/raw/ (traquees par DVC).
#
# Fonctions principales :
#   - load_consumption(path) -> DataFrame indexe par Mois (col: Consommation)
#   - load_dju(path) -> DataFrame indexe par Mois (col: DJU)
#   - load_merged(conso_path, dju_path) -> DataFrame fusionne (inner join sur Mois)
#
# Dependances externes : pandas
# ============================================================
import pandas as pd

DEFAULT_CONSO_PATH = "data/raw/consommation_mensuelle.csv"
DEFAULT_DJU_PATH = "data/raw/dju_mensuel.csv"


def load_consumption(path: str = DEFAULT_CONSO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Mois"])
    df = df[["Mois", "Consommation"]].dropna()
    df = df.set_index("Mois").sort_index()
    return df


def load_dju(path: str = DEFAULT_DJU_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Mois"])
    df = df[["Mois", "DJU"]].dropna()
    df = df.set_index("Mois").sort_index()
    return df


def load_merged(conso_path: str = DEFAULT_CONSO_PATH, dju_path: str = DEFAULT_DJU_PATH) -> pd.DataFrame:
    df_conso = load_consumption(conso_path)
    df_dju = load_dju(dju_path)
    return df_conso.join(df_dju, how="inner")
