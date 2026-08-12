# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : validation pydantic du dataframe fusionne (Consommation +
# DJU) au point d'entree du pipeline (data_loader.load_merged),
# avant qu'aucune donnee — France ou tout pays ajoute via
# connectors/ — n'atteigne l'entrainement. Inspire du pattern
# "valider aux frontieres d'ingestion" (Made-With-ML).
#
# Fonctions principales :
#   - validate_merged_dataframe(df, country, min_rows) : leve
#     DataQualityError listant TOUTES les violations trouvees
#     (pas d'arret a la premiere ligne invalide)
#
# Dependances externes : pydantic, pandas
# ============================================================
from typing import List

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator


class DataQualityError(ValueError):
    """Raised when a merged consumption/DJU dataframe fails validation."""


class MonthlyObservation(BaseModel):
    consommation: float
    dju: float

    @field_validator("consommation")
    @classmethod
    def _consommation_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Consommation must be > 0")
        return v

    @field_validator("dju")
    @classmethod
    def _dju_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("DJU must be >= 0")
        return v


def validate_merged_dataframe(df: pd.DataFrame, country: str = "france", min_rows: int = 24) -> None:
    """Validate a merged (Mois-indexed) Consommation/DJU dataframe.

    Checks row count, monotonic/unique/gap-free monthly index, and per-row value
    ranges via MonthlyObservation. Collects every violation and raises a single
    DataQualityError listing all of them, so a failed /train job is fully diagnostic.
    """
    errors: List[str] = []

    if len(df) < min_rows:
        errors.append(f"only {len(df)} rows, need at least {min_rows}")

    if not df.index.is_monotonic_increasing:
        errors.append("Mois index is not sorted ascending")

    if df.index.has_duplicates:
        dupes = sorted(str(m) for m in df.index[df.index.duplicated()].unique())
        errors.append(f"duplicate months: {dupes}")

    if len(df.index) > 1:
        actual_months = set(pd.DatetimeIndex(df.index).to_period("M"))
        expected_months = set(
            pd.period_range(min(actual_months), max(actual_months), freq="M")
        )
        missing = sorted(str(m) for m in expected_months - actual_months)
        if missing:
            shown = missing[:5]
            suffix = "..." if len(missing) > 5 else ""
            errors.append(f"missing months: {shown}{suffix}")

    for mois, row in df.iterrows():
        label = mois.date() if hasattr(mois, "date") else mois
        try:
            MonthlyObservation(consommation=row["Consommation"], dju=row["DJU"])
        except ValidationError as exc:
            errors.append(f"{label}: {exc.errors()[0]['msg']}")

    if errors:
        raise DataQualityError(
            f"[{country}] merged dataframe failed validation ({len(errors)} issue(s)): "
            + "; ".join(errors)
        )
