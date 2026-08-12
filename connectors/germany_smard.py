"""SMARD (Bundesnetzagentur) — German grid load, no API key at all. Filter 410 = total
"Realisierter Stromverbrauch" (realized consumption), hourly resolution.

https://github.com/bundesAPI/smard-api (community-documented, no official OpenAPI spec)
"""
import pandas as pd
import requests

from .base import CountryConnector

FILTER_CONSUMPTION = 410
REGION = "DE"
RESOLUTION = "hour"  # both the index and per-chunk filenames use the English resolution name
INDEX_URL = f"https://www.smard.de/app/chart_data/{FILTER_CONSUMPTION}/{REGION}/index_{RESOLUTION}.json"
CHUNK_URL = f"https://www.smard.de/app/chart_data/{FILTER_CONSUMPTION}/{REGION}/{FILTER_CONSUMPTION}_{REGION}_{RESOLUTION}_{{ts}}.json"


class SmardConnector(CountryConnector):
    country_code = "germany"
    api_key_env = None  # no auth required

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """start/end: 'YYYY-MM'. SMARD publishes weekly JSON chunks of hourly MWh values;
        this fetches every chunk overlapping [start, end] and aggregates to monthly GWh."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)

        index_resp = requests.get(INDEX_URL, timeout=30)
        index_resp.raise_for_status()
        all_timestamps = index_resp.json()["timestamps"]

        rows = []
        for ts in all_timestamps:
            chunk_start = pd.Timestamp(ts, unit="ms")
            if chunk_start < start_ts - pd.Timedelta(days=8) or chunk_start > end_ts:
                continue
            chunk_resp = requests.get(CHUNK_URL.format(ts=ts), timeout=30)
            chunk_resp.raise_for_status()
            for point_ts_ms, value_mwh in chunk_resp.json()["series"]:
                if value_mwh is None:
                    continue
                rows.append((pd.Timestamp(point_ts_ms, unit="ms"), value_mwh))

        df = pd.DataFrame(rows, columns=["timestamp", "value_mwh"])
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        df["Mois"] = df["timestamp"].values.astype("datetime64[M]")
        monthly = df.groupby("Mois", as_index=False)["value_mwh"].sum()
        monthly["Consommation"] = monthly["value_mwh"] / 1000.0  # MWh -> GWh
        return monthly[["Mois", "Consommation"]].sort_values("Mois").reset_index(drop=True)
