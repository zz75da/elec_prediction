# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : resolution de la configuration par pays a partir de
# params.yaml["data"]["countries"] — cote "lecture" du registre
# config-driven inspire du Data Catalog de Kedro (un code pays ->
# un bloc de config, plutot que des chemins en dur repetes dans
# chaque service).
#
# Fonctions principales :
#   - resolve_country_config(params, country) -> dict : bloc de
#     config du pays (ou default_country si country est None),
#     leve UnknownCountryError si le code est inconnu
#   - list_countries(params) -> list[{"code","label"}] : pour
#     GET /countries (peuple le selecteur Streamlit)
#
# Dependances externes : aucune (stdlib uniquement)
# ============================================================
from typing import List, Optional


class UnknownCountryError(KeyError):
    """Raised when a requested country code isn't in params.yaml's data.countries map."""


def resolve_country_config(params: dict, country: Optional[str] = None) -> dict:
    countries = params.get("data", {}).get("countries", {})
    resolved = country or params.get("default_country", "france")
    if resolved not in countries:
        raise UnknownCountryError(
            f"Unknown country '{resolved}' — valid codes: {sorted(countries.keys())}"
        )
    cfg = dict(countries[resolved])
    cfg["code"] = resolved
    return cfg


def list_countries(params: dict) -> List[dict]:
    countries = params.get("data", {}).get("countries", {})
    return [{"code": code, "label": cfg.get("label", code)} for code, cfg in countries.items()]
