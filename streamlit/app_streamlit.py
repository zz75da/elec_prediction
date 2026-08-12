# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : interface Streamlit (port 8501) pour declencher un
# entrainement, visualiser le backtest (MAPE/RMSE par modele) et
# obtenir des previsions au-dela du dernier mois observe, pour un
# pays choisi parmi ceux configures dans train-api/params.yaml
# (France + tout pays ajoute via connectors/). Appelle train-api
# et predict-api via HTTP. Pattern repris de
# rakuten_mlops_services/streamlit/app_streamlit.py (UI simple,
# pas d'auth ici — cf. decision de perimetre "stack allegee, pas
# de JWT").
#
# Pages :
#   - "Entraînement" : bouton POST /train {country}, affichage du statut/MAPE
#   - "Prévision" : formulaire horizon (+ DJU prévisionnel optionnel),
#     graphique historique + prévision + intervalle de confiance
#   - "Données historiques" : courbe Consommation / DJU brute (pays selectionne)
#   - "Comparaison 2019" : les 5 pays cote a cote sur leur seule annee civile
#     commune complete — conso totale, conso/habitant, profil saisonnier,
#     sensibilite a la temperature (pente + R² de Consommation ~ DJU)
#
# Le pays selectionne dans la barre laterale est garde dans
# st.session_state["country"] et reutilise sur les pages "Entraînement" /
# "Prévision" / "Données historiques". "Comparaison 2019" ignore ce choix —
# elle affiche systematiquement tous les pays disponibles.
#
# Dependances externes : streamlit, requests, pandas, numpy, plotly
# ============================================================
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

TRAIN_API_URL = os.getenv("TRAIN_API_URL", "http://localhost:5010")
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:5011")

st.set_page_config(page_title="elec_prediction", layout="wide")
st.title("⚡ Prévision de la consommation électrique")
st.caption("Correction température (DJU) + Holt-Winters / SARIMA(0,1,1)(1,1,1)₁₂")


@st.cache_data(ttl=60)
def _fetch_countries():
    resp = requests.get(f"{TRAIN_API_URL}/countries", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    _countries_payload = _fetch_countries()
    _country_options = {c["code"]: c["label"] for c in _countries_payload["countries"]}
    _default_country = _countries_payload.get("default_country", "france")
except requests.RequestException:
    _country_options = {"france": "France"}
    _default_country = "france"

if "country" not in st.session_state or st.session_state["country"] not in _country_options:
    st.session_state["country"] = _default_country

country = st.sidebar.selectbox(
    "Pays",
    options=list(_country_options.keys()),
    format_func=lambda code: _country_options.get(code, code),
    key="country",
)

page = st.sidebar.radio("Page", ["Entraînement", "Prévision", "Données historiques", "Comparaison 2019"])

# ---------------------------------------------------------------------------
if page == "Entraînement":
    st.header(f"Entraînement du modèle — {_country_options.get(country, country)}")
    st.write("Lance la pipeline: régression OLS (Consommation ~ DJU) → désaisonnalisation → "
             "Holt-Winters + SARIMA → backtest sur l'année de test → sélection du meilleur modèle.")

    if st.button("Lancer l'entraînement", type="primary"):
        try:
            resp = requests.post(f"{TRAIN_API_URL}/train", json={"country": country}, timeout=10)
            if resp.status_code == 400:
                st.error(f"Pays invalide: {resp.json().get('detail')}")
            else:
                resp.raise_for_status()
                job = resp.json()
                job_id = job["job_id"]
                st.info(f"Job lancé: `{job_id}`")

                progress = st.empty()
                with st.spinner("Entraînement en cours..."):
                    for _ in range(120):  # up to ~2 min polling
                        time.sleep(1)
                        status = requests.get(f"{TRAIN_API_URL}/train/status/{job_id}", timeout=10).json()
                        if status["status"] in ("success", "failed"):
                            break
                        progress.text(f"status: {status['status']}...")

                progress.empty()

                if status["status"] == "success":
                    st.success(f"Modèle retenu: **{status['best_model']}**")
                    col1, col2 = st.columns(2)
                    col1.metric("Holt-Winters MAPE", f"{status['metrics']['holt_winters']['mape']:.2f}%")
                    col2.metric("SARIMA MAPE", f"{status['metrics']['sarima']['mape']:.2f}%")
                    st.json(status)

                    try:
                        reload_resp = requests.post(
                            f"{PREDICT_API_URL}/reload-artifacts", params={"country": country}, timeout=10,
                        )
                        reload_resp.raise_for_status()
                        st.info("predict-api rechargé avec le nouveau modèle.")
                    except requests.RequestException as e:
                        st.warning(f"Entraînement réussi mais impossible de recharger predict-api ({PREDICT_API_URL}): {e}")
                else:
                    st.error(f"Échec: {status.get('error')}")
        except requests.RequestException as e:
            st.error(f"Impossible de contacter train-api ({TRAIN_API_URL}): {e}")

# ---------------------------------------------------------------------------
elif page == "Prévision":
    st.header(f"Prévision — {_country_options.get(country, country)}")
    horizon = st.slider("Horizon (mois)", min_value=1, max_value=36, value=12)
    use_dju = st.checkbox("Fournir un DJU prévisionnel (pour reconstruire la consommation réelle)")
    dju_forecast = None
    if use_dju:
        dju_text = st.text_input(f"DJU prévisionnel ({horizon} valeurs séparées par des virgules)")
        if dju_text:
            try:
                dju_forecast = [float(x.strip()) for x in dju_text.split(",")]
            except ValueError:
                st.warning("Format invalide — utilisez des nombres séparés par des virgules.")

    if st.button("Prévoir", type="primary"):
        payload = {"horizon": horizon, "country": country}
        if dju_forecast:
            payload["dju_forecast"] = dju_forecast
        try:
            resp = requests.post(f"{PREDICT_API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 503:
                st.warning(f"Aucun modèle entraîné pour {_country_options.get(country, country)} — lancez d'abord un entraînement.")
            else:
                resp.raise_for_status()
                result = resp.json()
                st.write(f"Modèle utilisé: **{result['model_used']}** · Dernier mois observé: {result['last_observed_month']}")

                df = pd.DataFrame(result["forecast"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["month"], y=df["conso_correction_pred_gwh"], name="Consommation corrigée (prévue)", mode="lines+markers"))
                if "ci_lower_gwh" in df.columns:
                    fig.add_trace(go.Scatter(x=df["month"], y=df["ci_upper_gwh"], line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=df["month"], y=df["ci_lower_gwh"], fill="tonexty", line=dict(width=0), name="Intervalle 95%"))
                if "consommation_pred_gwh" in df.columns:
                    fig.add_trace(go.Scatter(x=df["month"], y=df["consommation_pred_gwh"], name="Consommation réelle (reconstruite)", mode="lines+markers"))
                fig.update_layout(xaxis_title="Mois", yaxis_title="GWh")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)
        except requests.RequestException as e:
            st.error(f"Impossible de contacter predict-api ({PREDICT_API_URL}): {e}")

# ---------------------------------------------------------------------------
elif page == "Données historiques":
    st.header(f"Données historiques — {_country_options.get(country, country)}")
    try:
        conso = pd.read_csv(f"data/raw/{country}/consommation_mensuelle.csv", parse_dates=["Mois"])
        dju = pd.read_csv(f"data/raw/{country}/dju_mensuel.csv", parse_dates=["Mois"])
        merged = conso.merge(dju, on="Mois").sort_values("Mois").reset_index(drop=True)

        attribution_path = f"data/raw/{country}/ATTRIBUTION.txt"
        if os.path.exists(attribution_path):
            with open(attribution_path) as f:
                st.caption(f.read().strip())

        # Bounds come from this country's own data — France stops at 2019, the others
        # run later, so the slider's range is only ever what's actually available.
        min_date, max_date = merged["Mois"].min().date(), merged["Mois"].max().date()
        if min_date < max_date:
            date_range = st.slider(
                "Plage de dates", min_value=min_date, max_value=max_date,
                value=(min_date, max_date), format="YYYY-MM", key=f"date_range_{country}",
            )
            merged = merged[(merged["Mois"].dt.date >= date_range[0]) & (merged["Mois"].dt.date <= date_range[1])]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["Mois"], y=merged["Consommation"], name="Consommation (GWh)"))
        fig.add_trace(go.Scatter(x=merged["Mois"], y=merged["DJU"], name="DJU", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Consommation (GWh)"),
            yaxis2=dict(title="DJU", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged, use_container_width=True)
    except FileNotFoundError:
        st.warning(f"data/raw/{country}/*.csv introuvable — ce pays n'a pas encore été synchronisé "
                   f"(voir scripts/sync_country_data.py) ou le dossier data/ n'est pas monté (voir docker-compose.yml).")

# ---------------------------------------------------------------------------
else:  # page == "Comparaison 2019"
    st.header("Comparaison 2019 — tous les pays")
    st.write(
        "2019 est la seule année civile où les 5 pays disposent tous d'une série complète "
        "(France : 2009-2019 · les 4 autres : 2015/2019-2024) — c'est donc la seule base "
        "de comparaison directe entre pays."
    )

    # Fixed hue per country (never remapped) — same order as connectors/registry.py.
    # A country added later without a slot here falls back to neutral gray rather than crash.
    COUNTRY_COLORS = {
        "france": "#2a78d6",   # blue
        "usa": "#eb6834",      # orange
        "germany": "#1baf7a",  # aqua
        "uk": "#eda100",       # yellow
        "finland": "#e87ba4",  # magenta
    }
    GRID_COLOR = "#e1e0d9"
    FALLBACK_COLOR = "#898781"

    # World Bank 2019 population estimates (millions, rounded) — the only external
    # figures on this page; everything else is computed from data/raw/*.
    POPULATION_M = {"france": 67.4, "germany": 83.1, "usa": 328.2, "uk": 66.8, "finland": 5.52}
    MONTHS_FR = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    @st.cache_data(ttl=300)
    def _load_2019_comparison(codes: tuple):
        """Returns (summary_df keyed by country code, {code: monthly % profile}, {code: 2019 df}, missing codes)."""
        rows = []
        profiles = {}
        raw = {}
        missing = []
        for c in codes:
            try:
                conso = pd.read_csv(f"data/raw/{c}/consommation_mensuelle.csv", parse_dates=["Mois"])
                dju = pd.read_csv(f"data/raw/{c}/dju_mensuel.csv", parse_dates=["Mois"])
            except FileNotFoundError:
                missing.append(c)
                continue
            df = conso.merge(dju, on="Mois")
            df = df[df["Mois"].dt.year == 2019].sort_values("Mois").reset_index(drop=True)
            if len(df) < 12 or c not in POPULATION_M:
                missing.append(c)
                continue

            total = df["Consommation"].sum()
            pop = POPULATION_M[c]
            per_capita_mwh = (total * 1000) / (pop * 1e6)  # GWh -> MWh, then / persons
            peak_idx, trough_idx = df["Consommation"].idxmax(), df["Consommation"].idxmin()
            peak_trough_ratio = df["Consommation"].max() / df["Consommation"].min()

            slope, intercept = np.polyfit(df["DJU"], df["Consommation"], 1)
            r2 = np.corrcoef(df["DJU"], df["Consommation"])[0, 1] ** 2

            profile_pct = (df["Consommation"] / total * 100).values
            profiles[c] = profile_pct
            raw[c] = df

            rows.append({
                "code": c,
                "total_gwh": total,
                "population_m": pop,
                "per_capita_mwh": per_capita_mwh,
                "peak_month": df.loc[peak_idx, "Mois"].strftime("%b"),
                "trough_month": df.loc[trough_idx, "Mois"].strftime("%b"),
                "peak_trough_ratio": peak_trough_ratio,
                "dju_slope": slope,
                "dju_r2": r2,
                "winter_share_pct": profile_pct[[0, 1, 11]].sum(),
                "summer_share_pct": profile_pct[[5, 6, 7]].sum(),
            })
        return pd.DataFrame(rows), profiles, raw, missing

    summary, profiles, raw_2019, missing = _load_2019_comparison(tuple(sorted(_country_options.keys())))

    if missing:
        st.info(
            "Non inclus (année 2019 incomplète ou pays non synchronisé) : "
            + ", ".join(_country_options.get(c, c) for c in missing)
            + " — voir `scripts/sync_country_data.py`."
        )

    if summary.empty:
        st.error("Aucun pays ne dispose d'une année 2019 complète pour l'instant.")
    else:
        summary["Pays"] = summary["code"].map(lambda c: _country_options.get(c, c))
        colors = [COUNTRY_COLORS.get(c, FALLBACK_COLOR) for c in summary["code"]]

        # --- Headline KPIs ---
        top_capita = summary.loc[summary["per_capita_mwh"].idxmax()]
        low_capita = summary.loc[summary["per_capita_mwh"].idxmin()]
        top_r2 = summary.loc[summary["dju_r2"].idxmax()]
        summer_peaking = summary[summary["summer_share_pct"] > summary["winter_share_pct"]]

        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Conso. par habitant la plus élevée",
            top_capita["Pays"],
            f"{top_capita['per_capita_mwh']:.1f} MWh/hab — {top_capita['per_capita_mwh'] / low_capita['per_capita_mwh']:.1f}× {low_capita['Pays']}",
        )
        k2.metric("Le plus sensible à la température", top_r2["Pays"], f"R² = {top_r2['dju_r2']:.2f}")
        if not summer_peaking.empty:
            k3.metric("Seul(s) pays à pic estival", ", ".join(summer_peaking["Pays"]), "clim > chauffage")
        else:
            k3.metric("Profil saisonnier", "Tous à pic hivernal", "chauffage > clim, partout")

        st.markdown("---")

        # --- Total consumption (raw scale) ---
        st.subheader("Consommation totale 2019")
        fig1 = go.Figure(go.Bar(
            x=summary["Pays"], y=summary["total_gwh"], marker_color=colors,
            text=summary["total_gwh"].apply(lambda v: f"{v / 1000:,.0f} TWh"), textposition="outside",
        ))
        fig1.update_layout(yaxis_title="GWh", showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Échelle brute — reflète surtout la taille du pays, pas l'intensité énergétique par habitant "
                   "(graphique suivant).")

        # --- Per-capita consumption ---
        st.subheader("Consommation par habitant 2019")
        fig2 = go.Figure(go.Bar(
            x=summary["Pays"], y=summary["per_capita_mwh"], marker_color=colors,
            text=summary["per_capita_mwh"].apply(lambda v: f"{v:.1f}"), textposition="outside",
        ))
        fig2.update_layout(yaxis_title="MWh / habitant", showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Population 2019 (Banque Mondiale, arrondie) : " + ", ".join(
            f"{_country_options.get(c, c)} {POPULATION_M[c]:.1f}M" for c in summary["code"]
        ))

        # --- Seasonality profile ---
        st.subheader("Profil saisonnier — part de chaque mois dans le total annuel")
        fig3 = go.Figure()
        for c in summary["code"]:
            fig3.add_trace(go.Scatter(
                x=MONTHS_FR, y=profiles[c], name=_country_options.get(c, c), mode="lines+markers",
                line=dict(color=COUNTRY_COLORS.get(c, FALLBACK_COLOR), width=2), marker=dict(size=8),
            ))
        fig3.update_layout(
            yaxis_title="% du total annuel", yaxis=dict(gridcolor=GRID_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, use_container_width=True)
        if not summer_peaking.empty:
            st.info(
                f"**{', '.join(summer_peaking['Pays'])}** est le seul profil où l'été (Jun+Jul+Aoû, "
                f"{summer_peaking.iloc[0]['summer_share_pct']:.1f}% du total) pèse plus lourd que l'hiver "
                f"(Déc+Jan+Fév, {summer_peaking.iloc[0]['winter_share_pct']:.1f}%) — signature typique d'une "
                f"demande tirée par la climatisation plutôt que par le chauffage électrique."
            )

        # --- Temperature sensitivity: small multiples (one panel per country, own scale) ---
        st.subheader("Sensibilité à la température (Consommation vs DJU, 2019)")
        codes = list(summary["code"])
        fig4 = make_subplots(rows=1, cols=len(codes), subplot_titles=[
            f"{_country_options.get(c, c)} (R²={summary.set_index('code').loc[c, 'dju_r2']:.2f})" for c in codes
        ])
        for i, c in enumerate(codes, start=1):
            df = raw_2019[c]
            color = COUNTRY_COLORS.get(c, FALLBACK_COLOR)
            fig4.add_trace(go.Scatter(x=df["DJU"], y=df["Consommation"], mode="markers",
                                       marker=dict(color=color, size=8), showlegend=False), row=1, col=i)
            slope, intercept = np.polyfit(df["DJU"], df["Consommation"], 1)
            x_line = np.array([df["DJU"].min(), df["DJU"].max()])
            fig4.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                       line=dict(color=color, width=2, dash="dot"), showlegend=False), row=1, col=i)
            fig4.update_xaxes(title_text="DJU", row=1, col=i, gridcolor=GRID_COLOR)
            fig4.update_yaxes(title_text="GWh" if i == 1 else None, row=1, col=i, gridcolor=GRID_COLOR)
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "Chaque panneau a sa propre échelle — l'objectif est de comparer la pente et la dispersion des "
            "points, pas les niveaux absolus. R² proche de 1 : la température seule explique presque toute "
            "la variation mensuelle (chauffage électrique dominant). R² faible : d'autres facteurs "
            "(activité économique, industrie lourde, mix de chauffage non-électrique) dominent davantage."
        )

        # --- Summary table (accessible fallback for every chart above) ---
        st.subheader("Tableau récapitulatif")
        display_cols = {
            "Pays": "Pays", "total_gwh": "Total (GWh)", "per_capita_mwh": "Par habitant (MWh)",
            "peak_month": "Mois de pointe", "trough_month": "Mois creux",
            "peak_trough_ratio": "Ratio pointe/creux", "dju_r2": "Sensibilité DJU (R²)",
            "winter_share_pct": "Part hiver %", "summer_share_pct": "Part été %",
        }
        st.dataframe(
            summary[list(display_cols.keys())].rename(columns=display_cols).round(2),
            use_container_width=True, hide_index=True,
        )
