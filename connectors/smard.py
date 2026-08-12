"""SMARD (Bundesnetzagentur transparency platform) — grid load for Germany, Austria, and
Luxembourg, no API key at all for any of them. Filter 410 = total "Realisierter
Stromverbrauch" (realized consumption), hourly resolution. One connector class,
parameterized by SMARD's region code, since all three countries publish through the same
platform (SMARD historically covered the joint DE-AT-LU bidding zone before Austria split
out in 2018 — the platform still serves each country's data individually).

https://github.com/bundesAPI/smard-api (community-documented, no official OpenAPI spec)
Licence: Datenlizenz Deutschland – Namensnennung 2.0 (dl-de/by-2.0) — requires attribution.
"""
import pandas as pd
import requests

from .base import CountryConnector

FILTER_CONSUMPTION = 410
RESOLUTION = "hour"  # both the index and per-chunk filenames use the English resolution name

ATTRIBUTION = "Contains data from SMARD.de (Bundesnetzagentur), licensed under the Datenlizenz Deutschland – Namensnennung 2.0 (dl-de/by-2.0)"


class SmardConnector(CountryConnector):
    api_key_env = None  # no auth required for any SMARD region
    attribution = ATTRIBUTION

    def __init__(self, region: str, country_code: str, api_key=None):
        super().__init__(api_key=api_key)
        self.region = region
        self.country_code = country_code

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """start/end: 'YYYY-MM'. SMARD publishes weekly JSON chunks of hourly MWh values;
        this fetches every chunk overlapping [start, end] and aggregates to monthly GWh."""
        index_url = f"https://www.smard.de/app/chart_data/{FILTER_CONSUMPTION}/{self.region}/index_{RESOLUTION}.json"
        chunk_url_tpl = (
            f"https://www.smard.de/app/chart_data/{FILTER_CONSUMPTION}/{self.region}/"
            f"{FILTER_CONSUMPTION}_{self.region}_{RESOLUTION}_{{ts}}.json"
        )

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.offsets.MonthEnd(0)

        index_resp = requests.get(index_url, timeout=30)
        index_resp.raise_for_status()
        all_timestamps = index_resp.json()["timestamps"]

        rows = []
        for ts in all_timestamps:
            chunk_start = pd.Timestamp(ts, unit="ms")
            if chunk_start < start_ts - pd.Timedelta(days=8) or chunk_start > end_ts:
                continue
            chunk_resp = requests.get(chunk_url_tpl.format(ts=ts), timeout=30)
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
