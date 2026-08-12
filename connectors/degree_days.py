"""Daily mean temperature -> monthly heating-degree-days (DJU), same convention as the
existing French "Degres Jours Unifies" series: HDD_day = max(base_temp_c - temp, 0),
summed per calendar month.
"""
import pandas as pd


def daily_temps_to_monthly_dju(df_daily: pd.DataFrame, base_temp_c: float = 18.0) -> pd.DataFrame:
    """df_daily: columns ['date', 'temperature_mean_c']. Returns columns ['Mois', 'DJU']."""
    df = df_daily.copy()
    df["hdd"] = (base_temp_c - df["temperature_mean_c"]).clip(lower=0)
    df["Mois"] = pd.to_datetime(df["date"]).values.astype("datetime64[M]")
    monthly = df.groupby("Mois", as_index=False)["hdd"].sum().rename(columns={"hdd": "DJU"})
    return monthly
