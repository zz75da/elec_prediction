"""Sync a country's electricity consumption + temperature-derived DJU into
data/raw/<country>/, ready for train-api to read exactly like France's committed CSVs.

Host-run only (never called from inside the train-api/predict-api containers — see
connectors/base.py's docstring for why). Loads API keys from .env via python-dotenv.

Usage:
    python scripts/sync_country_data.py --country usa --start 2015-01 --end 2024-12
    python scripts/sync_country_data.py --country all  --start 2015-01 --end 2024-12
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "train-api"))
sys.path.insert(0, str(REPO_ROOT))

from connectors import degree_days, open_meteo, registry  # noqa: E402
from services import schemas  # noqa: E402


def sync_one(country: str, start: str, end: str) -> None:
    spec = registry.get_spec(country)

    if spec.connector_cls is None:
        raise SystemExit(
            f"'{country}' has no live connector — its data is a static historical export. "
            f"Use scripts/extract_raw_data.py instead."
        )

    if spec.api_key_env:
        import os
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise SystemExit(
                f"Missing {spec.api_key_env} — add it to .env (see .env.template) before syncing '{country}'."
            )
    else:
        api_key = None

    print(f"[{country}] Fetching consumption from {spec.connector_cls.__name__}...")
    connector = spec.connector_cls(api_key=api_key, **spec.connector_kwargs)
    conso_df = connector.fetch_consumption(start, end)

    start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    print(f"[{country}] Fetching daily temperature from Open-Meteo...")
    daily_temps = open_meteo.fetch_daily_mean_temperature(spec.latitude, spec.longitude, start_date, end_date)
    dju_df = degree_days.daily_temps_to_monthly_dju(daily_temps, base_temp_c=spec.degree_day_base_c)

    merged = conso_df.merge(dju_df, on="Mois", how="inner").set_index("Mois").sort_index()
    schemas.validate_merged_dataframe(merged, country=country)
    print(f"[{country}] Validation passed — {len(merged)} months, "
          f"{merged.index.min().date()} to {merged.index.max().date()}.")

    out_dir = REPO_ROOT / "data" / "raw" / country
    out_dir.mkdir(parents=True, exist_ok=True)

    conso_df.to_csv(out_dir / "consommation_mensuelle.csv", index=False)
    dju_df.to_csv(out_dir / "dju_mensuel.csv", index=False)
    merged.reset_index().to_csv(out_dir / "consommation_dju_mensuel.csv", index=False)

    if spec.attribution:
        (out_dir / "ATTRIBUTION.txt").write_text(spec.attribution.format(year=datetime.now().year) + "\n")

    print(f"[{country}] Wrote {out_dir}/consommation_mensuelle.csv, dju_mensuel.csv, consommation_dju_mensuel.csv")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True,
                         help="Country code (usa, germany, austria, luxembourg, uk, finland, or 'all')")
    parser.add_argument("--start", required=True, help="Start month, YYYY-MM")
    parser.add_argument("--end", required=True, help="End month, YYYY-MM")
    args = parser.parse_args()

    if args.country == "all":
        targets = [c for c, spec in registry.REGISTRY.items() if spec.connector_cls is not None]
    else:
        targets = [args.country]

    for country in targets:
        sync_one(country, args.start, args.end)


if __name__ == "__main__":
    main()
