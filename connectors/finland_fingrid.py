"""Fingrid Open Data — dataset 124, "Electricity consumption in Finland" (short-interval,
instantaneous MW readings — currently ~15 minutes, but each record carries its own
startTime/endTime so the interval is computed per-record rather than assumed). Free instant
API key (header x-api-key): https://data.fingrid.fi

CC BY 4.0.
"""
import time

import pandas as pd
import requests

from .base import CountryConnector

DATASET_ID = 124
DATA_URL = f"https://data.fingrid.fi/api/datasets/{DATASET_ID}/data"
PAGE_SIZE = 20000
REQUEST_DELAY_SECONDS = 1.5  # Fingrid's API rate-limits aggressive back-to-back pagination
MAX_RETRIES = 5


def _get_with_retry(url, params, headers):
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = float(resp.headers.get("Retry-After", 2 ** attempt * 2))
        time.sleep(wait)
    resp.raise_for_status()
    return resp


class FingridConnector(CountryConnector):
    country_code = "finland"
    api_key_env = "FINGRID_API_KEY"

    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """start/end: 'YYYY-MM'. Paginates through every reading in [start, end], converts each
        instantaneous MW value to energy using that record's own (endTime - startTime) interval,
        and aggregates to monthly GWh."""
        start_iso = pd.Timestamp(start).strftime("%Y-%m-%dT00:00:00Z")
        end_iso = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%dT23:59:59Z")
        headers = {"x-api-key": self.api_key}

        rows = []
        page = 1
        last_page = None  # only page 1's response includes `lastPage` — later pages omit it
        while True:
            resp = _get_with_retry(
                DATA_URL,
                params={"startTime": start_iso, "endTime": end_iso, "pageSize": PAGE_SIZE, "page": page},
                headers=headers,
            )
            payload = resp.json()
            rows.extend(payload["data"])
            if last_page is None:
                last_page = payload.get("pagination", {}).get("lastPage", page)
            if page >= last_page:
                break
            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)

        df = pd.DataFrame(rows)
        df["startTime"] = pd.to_datetime(df["startTime"])
        df["endTime"] = pd.to_datetime(df["endTime"])
        interval_hours = (df["endTime"] - df["startTime"]).dt.total_seconds() / 3600.0
        df["energy_mwh"] = df["value"].astype(float) * interval_hours
        df["Mois"] = df["startTime"].values.astype("datetime64[M]")
        monthly = df.groupby("Mois", as_index=False)["energy_mwh"].sum()
        monthly["Consommation"] = monthly["energy_mwh"] / 1000.0  # MWh -> GWh
        return monthly[["Mois", "Consommation"]].sort_values("Mois").reset_index(drop=True)
