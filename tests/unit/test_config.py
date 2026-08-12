"""Unit tests for train-api/services/config.py — country config resolution."""
import pytest

from services import config

PARAMS = {
    "default_country": "france",
    "data": {
        "countries": {
            "france": {"label": "France", "raw_conso_path": "a.csv", "raw_dju_path": "b.csv", "test_year": 2019},
            "usa": {"label": "United States", "raw_conso_path": "c.csv", "raw_dju_path": "d.csv", "test_year": 2019},
        }
    },
}


def test_resolve_known_country():
    cfg = config.resolve_country_config(PARAMS, "usa")
    assert cfg["code"] == "usa"
    assert cfg["label"] == "United States"


def test_resolve_none_uses_default_country():
    cfg = config.resolve_country_config(PARAMS, None)
    assert cfg["code"] == "france"


def test_resolve_unknown_country_raises():
    with pytest.raises(config.UnknownCountryError, match="Unknown country"):
        config.resolve_country_config(PARAMS, "atlantis")


def test_list_countries():
    countries = config.list_countries(PARAMS)
    codes = {c["code"] for c in countries}
    assert codes == {"france", "usa"}
