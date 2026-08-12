"""UK National Energy System Operator (NESO) — "Historic Demand Data" dataset, half-hourly
national demand (ND column, MW), no API key at all. One CSV per calendar year, discovered
dynamically via the CKAN `datapackage_show` API rather than hardcoded per-year URLs (the
per-resource UUIDs are not derivable from the year alone and can change).

https://www.neso.energy/data-portal/historic-demand-data
Licence: NESO Open Data Licence — https://www.neso.energy/data-portal/ngeso-open-licence
"""
import re
from io import StringIO

import pandas as pd
import requests

from .base import CountryConnector

DATASET_ID = "historic-demand-data"
DATAPACKAGE_URL = f"https://api.neso.energy/api/3/action/datapackage_show?id={DATASET_ID}"
RESOURCE_NAME_RE = re.compile(r"historic_demand_data_(\d{4})")

ATTRIBUTION = "Contains NESO data, https://www.neso.energy/data-portal/historic-demand-data, licensed under the NESO Open Data Licence"


class NesoConnector(CountryConnector):
    country_code = "uk"
    api_key_env = None  # no auth required
    attribution = ATTRIBUTION

    def _resource_urls_by_year(self) -> dict:
        resp = requests.get(DATAPACKAGE_URL, timeout=30)
        resp.raise_for_status()
        resources = resp.json()["result"]["resources"]
        by_year = {}
        for r in resources:
            match = RESOURCE_NAME_RE.fullmatch(r.get("name", ""))
            if match:
                by_year[int(match.group(1))] = r["path"]
        return by_year

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """start/end: 'YYYY-MM'. ND (national demand, MW) is reported per half-hourly
        settlement period; energy = ND * 0.5h, summed per month and converted MWh -> GWh."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)

        urls_by_year = self._resource_urls_by_year()
        frames = []
        for year in range(start_ts.year, end_ts.year + 1):
            url = urls_by_year.get(year)
            if url is None:
                continue
            csv_resp = requests.get(url, timeout=60)
            csv_resp.raise_for_status()
            df_year = pd.read_csv(StringIO(csv_resp.text))
            frames.append(df_year[["SETTLEMENT_DATE", "ND"]])

        df = pd.concat(frames, ignore_index=True)
        # Older years' CSVs use a different date format than recent ones — "mixed" handles both.
        df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"], format="mixed", dayfirst=False)
        df = df[(df["SETTLEMENT_DATE"] >= start_ts) & (df["SETTLEMENT_DATE"] <= end_ts)]
        df["energy_mwh"] = df["ND"].astype(float) * 0.5  # half-hourly settlement period
        df["Mois"] = df["SETTLEMENT_DATE"].values.astype("datetime64[M]")
        monthly = df.groupby("Mois", as_index=False)["energy_mwh"].sum()
        monthly["Consommation"] = monthly["energy_mwh"] / 1000.0  # MWh -> GWh
        return monthly[["Mois", "Consommation"]].sort_values("Mois").reset_index(drop=True)
