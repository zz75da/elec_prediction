"""
One-time ETL that reproduces P9_01_notebook.ipynb cells 1, 5, 12-17.

Source files (NOT included in this repo — they live in the original
OpenClassrooms project folder and/or can be re-downloaded from RTE open data):
  - equilibre-national-mensuel-prod-conso-brute.csv
    https://opendata.reseaux-energies.fr/explore/dataset/equilibre-national-mensuel-prod-conso-brute/
  - DJU_meteo_chauf_ge.xlsx + calcul_DJU_10_05_2022_{a..g}.xlsx
    (8 French regions, Degrés Jours Unifiés — heating degree days, skiprows=11 header block)

Outputs (already committed under data/raw/, DVC-tracked):
  - data/raw/consommation_mensuelle.csv
  - data/raw/dju_mensuel.csv
  - data/raw/consommation_dju_mensuel.csv   (merged, for reference/tests)

Run again only if you have the original source files and want to refresh
the raw data (e.g. extend the series past 2019-12):

    python scripts/extract_raw_data.py --source-dir /path/to/original/OC_projets/folder
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw"

START = datetime.strptime("2008-12-01", "%Y-%m-%d")
END = datetime.strptime("2020-01-01", "%Y-%m-%d")

REGION_FILES = [
    "DJU_meteo_chauf_ge.xlsx",
    "Data/calcul_DJU_10_05_2022_a.xlsx",
    "Data/calcul_DJU_10_05_2022_b.xlsx",
    "Data/calcul_DJU_10_05_2022_c.xlsx",
    "Data/calcul_DJU_10_05_2022_d.xlsx",
    "Data/calcul_DJU_10_05_2022_e.xlsx",
    "Data/calcul_DJU_10_05_2022_f.xlsx",
    "Data/calcul_DJU_10_05_2022_g.xlsx",
]

MONTH_MAP = {
    "Unnamed: 0": "Annee", "JAN": "01", "FÉV": "02", "MAR": "03", "AVR": "04",
    "MAI": "05", "JUN": "06", "JUI": "07", "AOÛ": "08", "SEP": "09",
    "OCT": "10", "NOV": "11", "DÉC": "12",
}


def extract_consumption(source_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(source_dir / "equilibre-national-mensuel-prod-conso-brute.csv", sep=";")
    df = df[["Mois", "Consommation brute (GWh)"]]
    df["Mois"] = pd.to_datetime(df["Mois"])
    df = df.loc[(df["Mois"] > START) & (df["Mois"] < END), :]
    df = df.rename(columns={"Consommation brute (GWh)": "Consommation"})
    return df.sort_values("Mois")


def extract_dju(source_dir: Path) -> pd.DataFrame:
    frames = [pd.read_excel(source_dir / f, skiprows=11) for f in REGION_FILES]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=["Total"]).rename(columns=MONTH_MAP)

    df_mean = df.groupby("Annee", as_index=False).mean(numeric_only=True)
    df_mean = df_mean[df_mean["Annee"] != 2021]  # incomplete year in the source
    df_mean = round(df_mean, 1)

    long = pd.melt(df_mean, id_vars=["Annee"], var_name="Mois", value_name="DJU")
    long["Mois"] = long["Annee"].astype(int).astype(str) + "-" + long["Mois"] + "-01"
    long = long.drop(columns=["Annee"])
    long["Mois"] = pd.to_datetime(long["Mois"], format="%Y-%m-%d")
    long = long.loc[(long["Mois"] > START) & (long["Mois"] < datetime.strptime("2020-10-01", "%Y-%m-%d")), :]
    return long.sort_values("Mois")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True,
                         help="Path to the folder containing the original xlsx/csv source files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_conso = extract_consumption(args.source_dir)
    df_conso.to_csv(OUT_DIR / "consommation_mensuelle.csv", index=False)
    print(f"Consommation: {len(df_conso)} lignes -> {OUT_DIR / 'consommation_mensuelle.csv'}")

    df_dju = extract_dju(args.source_dir)
    df_dju.to_csv(OUT_DIR / "dju_mensuel.csv", index=False)
    print(f"DJU: {len(df_dju)} lignes -> {OUT_DIR / 'dju_mensuel.csv'}")

    merged = pd.merge(df_conso, df_dju, on="Mois")
    merged.to_csv(OUT_DIR / "consommation_dju_mensuel.csv", index=False)
    print(f"Merged: {len(merged)} lignes -> {OUT_DIR / 'consommation_dju_mensuel.csv'}")


if __name__ == "__main__":
    main()
