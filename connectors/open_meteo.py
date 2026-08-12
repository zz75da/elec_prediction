"""Free, no-key historical daily temperature — shared by every country connector's
degree-day computation so the DJU pipeline stays consistent across countries.

https://open-meteo.com/en/docs/historical-weather-api
"""
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily_mean_temperature(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame with columns ['date', 'temperature_mean_c'] for [start, end] (YYYY-MM-DD)."""
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean",
            "timezone": "UTC",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(payload["time"]),
        "temperature_mean_c": payload["temperature_2m_mean"],
    })
