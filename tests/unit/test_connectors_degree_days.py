"""Unit test for connectors/degree_days.py — pure function, no network."""
import pandas as pd

from connectors.degree_days import daily_temps_to_monthly_dju


def test_daily_temps_to_monthly_dju_hand_computed():
    # January: 3 days at 10, 15, 20 degrees C, base 18 -> HDD = 8, 3, 0 -> sum = 11
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "temperature_mean_c": [10.0, 15.0, 20.0],
    })
    monthly = daily_temps_to_monthly_dju(daily, base_temp_c=18.0)
    assert len(monthly) == 1
    assert monthly.loc[0, "Mois"] == pd.Timestamp("2024-01-01")
    assert monthly.loc[0, "DJU"] == 11.0


def test_daily_temps_to_monthly_dju_no_negative_hdd():
    # All days warmer than base -> HDD clipped to 0, DJU sums to 0
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
        "temperature_mean_c": [25.0, 30.0],
    })
    monthly = daily_temps_to_monthly_dju(daily, base_temp_c=18.0)
    assert monthly.loc[0, "DJU"] == 0.0


def test_daily_temps_to_monthly_dju_splits_across_months():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-02-01"]),
        "temperature_mean_c": [8.0, 8.0],  # HDD = 10 each day
    })
    monthly = daily_temps_to_monthly_dju(daily, base_temp_c=18.0).set_index("Mois")
    assert monthly.loc[pd.Timestamp("2024-01-01"), "DJU"] == 10.0
    assert monthly.loc[pd.Timestamp("2024-02-01"), "DJU"] == 10.0
