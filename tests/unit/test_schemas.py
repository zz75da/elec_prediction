"""Unit tests for train-api/services/schemas.py — the ingestion-boundary validation
used by data_loader.load_merged() for every country's data."""
import pandas as pd
import pytest

from services import schemas


def _valid_df(n=30):
    months = pd.date_range("2015-01-31", periods=n, freq="M")
    return pd.DataFrame({"Consommation": [40000.0] * n, "DJU": [200.0] * n}, index=months)


def test_valid_dataframe_passes():
    schemas.validate_merged_dataframe(_valid_df(), country="testcountry")  # no exception


def test_negative_consommation_raises():
    df = _valid_df()
    df.iloc[5, df.columns.get_loc("Consommation")] = -100.0
    with pytest.raises(schemas.DataQualityError, match="Consommation must be > 0"):
        schemas.validate_merged_dataframe(df, country="testcountry")


def test_negative_dju_raises():
    df = _valid_df()
    df.iloc[5, df.columns.get_loc("DJU")] = -1.0
    with pytest.raises(schemas.DataQualityError, match="DJU must be >= 0"):
        schemas.validate_merged_dataframe(df, country="testcountry")


def test_too_few_rows_raises():
    with pytest.raises(schemas.DataQualityError, match="need at least"):
        schemas.validate_merged_dataframe(_valid_df(n=5), country="testcountry", min_rows=24)


def test_missing_month_raises():
    df = _valid_df(n=30)
    df = df.drop(df.index[10])  # punch a gap in the middle of the series
    with pytest.raises(schemas.DataQualityError, match="missing months"):
        schemas.validate_merged_dataframe(df, country="testcountry")


def test_multiple_violations_all_reported():
    df = _valid_df()
    df.iloc[0, df.columns.get_loc("Consommation")] = -1.0
    df.iloc[1, df.columns.get_loc("DJU")] = -1.0
    with pytest.raises(schemas.DataQualityError) as exc_info:
        schemas.validate_merged_dataframe(df, country="testcountry")
    message = str(exc_info.value)
    assert "Consommation must be > 0" in message
    assert "DJU must be >= 0" in message
