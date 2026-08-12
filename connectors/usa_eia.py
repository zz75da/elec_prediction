"""US Energy Information Administration (EIA) API v2 — retail electricity sales, national
total, natively monthly. Free instant API key: https://www.eia.gov/opendata/

https://www.eia.gov/opendata/documentation.php
"""
import pandas as pd
import requests

from .base import CountryConnector

RETAIL_SALES_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"


class EiaConnector(CountryConnector):
    country_code = "usa"
    api_key_env = "EIA_API_KEY"

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """start/end: 'YYYY-MM'. EIA's 'sales' figure for retail-sales is in million kWh,
        which is numerically equal to GWh."""
        resp = requests.get(
            RETAIL_SALES_URL,
            params={
                "api_key": self.api_key,
                "frequency": "monthly",
                "data[]": "sales",
                "facets[stateid][]": "US",
                "facets[sectorid][]": "ALL",  # total across all sectors — omitting this returns
                                               # one row per sector (COM/IND/OTH/RES/TRA/ALL)
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "length": 5000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()["response"]["data"]
        df = pd.DataFrame(rows)
        df["Mois"] = pd.to_datetime(df["period"], format="%Y-%m")
        df["Consommation"] = df["sales"].astype(float)
        return df[["Mois", "Consommation"]].sort_values("Mois").reset_index(drop=True)
