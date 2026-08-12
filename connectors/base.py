"""Adapter-pattern interface for a country's electricity-consumption data source.

Host-run only — never imported by train-api/predict-api (see connectors/README or the
repo README's "Multi-country data" section for why). Each concrete connector fetches
consumption from its country's native API; temperature/degree-days is handled uniformly
for every connector via connectors.open_meteo + connectors.degree_days, so the OLS
Consommation~DJU pipeline in train-api never needs to know where the DJU came from.
"""
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class CountryConnector(ABC):
    country_code: str
    api_key_env: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    def fetch_consumption(self, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame with columns ['Mois', 'Consommation'] (GWh, monthly)."""
        raise NotImplementedError
