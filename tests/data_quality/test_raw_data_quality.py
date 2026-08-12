"""Data-quality CI stage — validates each configured country's committed raw CSVs,
offline, no network. Distinct from train-api/tests/test_model_quality.py (model
performance): this only checks the data feeding the pipeline is well-formed.

Skips (does not fail) a country whose raw CSV directory hasn't been synced yet — usa/uk/
finland need free API keys only the user can supply (see .env.template), via
scripts/sync_country_data.py. France and germany ship with committed data.
"""
from pathlib import Path

import pytest
import yaml

from services import data_loader

REPO_ROOT = Path(__file__).resolve().parents[2]

with open(REPO_ROOT / "params.yaml") as f:
    _PARAMS = yaml.safe_load(f)

_COUNTRIES = _PARAMS.get("data", {}).get("countries", {})


def _raw_files_present(cfg: dict) -> bool:
    return (REPO_ROOT / cfg["raw_conso_path"]).exists() and (REPO_ROOT / cfg["raw_dju_path"]).exists()


def _load(country: str, cfg: dict):
    # Validation (schemas.validate_merged_dataframe) happens inside load_merged itself —
    # a malformed CSV raises DataQualityError here, failing the test with the full report.
    return data_loader.load_merged(
        str(REPO_ROOT / cfg["raw_conso_path"]), str(REPO_ROOT / cfg["raw_dju_path"]), country=country,
    )


@pytest.mark.parametrize("country", sorted(_COUNTRIES.keys()))
def test_raw_data_passes_schema_validation(country):
    cfg = _COUNTRIES[country]
    if not _raw_files_present(cfg):
        pytest.skip(f"{country}: raw CSVs not synced yet (see scripts/sync_country_data.py)")
    df = _load(country, cfg)
    assert len(df) > 0


@pytest.mark.parametrize("country", sorted(_COUNTRIES.keys()))
def test_configured_test_year_is_within_range(country):
    cfg = _COUNTRIES[country]
    if not _raw_files_present(cfg):
        pytest.skip(f"{country}: raw CSVs not synced yet (see scripts/sync_country_data.py)")
    df = _load(country, cfg)
    test_year = cfg["test_year"]
    assert df.index.min().year <= test_year <= df.index.max().year, (
        f"{country}: quality_gate test_year={test_year} is outside the synced data range "
        f"[{df.index.min().year}, {df.index.max().year}]"
    )
