"""Config-driven connector lookup — one entry per country, chosen by code rather than
hardcoded branching. Adding a new country is one new connector module (or, for a platform
that already covers multiple countries like SMARD, one new registry entry reusing the
existing connector class via connector_kwargs) plus one entry here.

France is deliberately absent from CONNECTOR_CLS (connector_cls=None): its data is a static
historical export (scripts/extract_raw_data.py), not synced from a live API.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from .base import CountryConnector
from .finland_fingrid import FingridConnector
from .smard import SmardConnector
from .uk_neso import NesoConnector
from .usa_eia import EiaConnector

SMARD_ATTRIBUTION = (
    "Contains data from SMARD.de (Bundesnetzagentur), licensed under the "
    "Datenlizenz Deutschland – Namensnennung 2.0 (dl-de/by-2.0)"
)


@dataclass(frozen=True)
class ConnectorSpec:
    code: str
    label: str
    connector_cls: Optional[Type[CountryConnector]]
    latitude: float
    longitude: float
    degree_day_base_c: float = 18.0
    api_key_env: Optional[str] = None
    attribution: Optional[str] = None
    connector_kwargs: Dict[str, Any] = field(default_factory=dict)  # extra constructor args, e.g. SMARD's region


REGISTRY = {
    "france": ConnectorSpec(
        code="france", label="France", connector_cls=None,
        latitude=46.6, longitude=2.2,
    ),
    "usa": ConnectorSpec(
        code="usa", label="United States", connector_cls=EiaConnector,
        latitude=38.9, longitude=-77.0, api_key_env="EIA_API_KEY",  # Washington, DC
    ),
    "germany": ConnectorSpec(
        code="germany", label="Germany", connector_cls=SmardConnector,
        latitude=52.5, longitude=13.4,  # Berlin
        attribution=SMARD_ATTRIBUTION,
        connector_kwargs={"region": "DE", "country_code": "germany"},
    ),
    "austria": ConnectorSpec(
        code="austria", label="Austria", connector_cls=SmardConnector,
        latitude=48.2, longitude=16.4,  # Vienna
        attribution=SMARD_ATTRIBUTION,
        connector_kwargs={"region": "AT", "country_code": "austria"},
    ),
    "luxembourg": ConnectorSpec(
        code="luxembourg", label="Luxembourg", connector_cls=SmardConnector,
        latitude=49.6, longitude=6.1,  # Luxembourg City
        attribution=SMARD_ATTRIBUTION,
        connector_kwargs={"region": "LU", "country_code": "luxembourg"},
    ),
    "uk": ConnectorSpec(
        code="uk", label="United Kingdom", connector_cls=NesoConnector,
        latitude=51.5, longitude=-0.1,  # London — no API key needed (NESO Open Data Licence)
        attribution="Contains NESO data, https://www.neso.energy/data-portal/historic-demand-data, "
                     "licensed under the NESO Open Data Licence",
    ),
    "finland": ConnectorSpec(
        code="finland", label="Finland", connector_cls=FingridConnector,
        latitude=60.2, longitude=24.9, api_key_env="FINGRID_API_KEY",  # Helsinki
    ),
}


def get_spec(country: str) -> ConnectorSpec:
    if country not in REGISTRY:
        raise KeyError(f"Unknown country '{country}' — valid codes: {sorted(REGISTRY.keys())}")
    return REGISTRY[country]
