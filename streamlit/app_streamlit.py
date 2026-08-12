# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : interface Streamlit (port 8501) pour declencher un
# entrainement, visualiser le backtest 2019 (MAPE/RMSE par
# modele) et obtenir des previsions au-dela du dernier mois
# observe. Appelle train-api et predict-api via HTTP. Pattern
# repris de rakuten_mlops_services/streamlit/app_streamlit.py
# (UI simple, pas d'auth ici — cf. decision de perimetre
# "stack allegee, pas de JWT").
#
# Pages :
#   - "Entraînement" : bouton POST /train, affichage du statut/MAPE
#   - "Prévision" : formulaire horizon (+ DJU prévisionnel optionnel),
#     graphique historique + prévision + intervalle de confiance
#   - "Historique" : courbe Consommation / DJU brute
#
# Dependances externes : streamlit, requests, pandas, plotly
# ============================================================
import os
import time

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

TRAIN_API_URL = os.getenv("TRAIN_API_URL", "http://localhost:5010")
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:5011")

st.set_page_config(page_title="elec_prediction", layout="wide")
st.title("⚡ Prévision de la consommation électrique — France")
st.caption("Correction température (DJU) + Holt-Winters / SARIMA(0,1,1)(1,1,1)₁₂ — MAPE historique = 2.22% (2019)")

page = st.sidebar.radio("Page", ["Entraînement", "Prévision", "Données historiques"])

# ---------------------------------------------------------------------------
if page == "Entraînement":
    st.header("Entraînement du modèle")
    st.write("Lance la pipeline: régression OLS (Consommation ~ DJU) → désaisonnalisation → "
             "Holt-Winters + SARIMA → backtest sur l'année de test → sélection du meilleur modèle.")

    if st.button("Lancer l'entraînement", type="primary"):
        try:
            resp = requests.post(f"{TRAIN_API_URL}/train", json={}, timeout=10)
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

            if status["status"] == "success":
                st.success(f"Modèle retenu: **{status['best_model']}**")
                col1, col2 = st.columns(2)
                col1.metric("Holt-Winters MAPE", f"{status['metrics']['holt_winters']['mape']:.2f}%")
                col2.metric("SARIMA MAPE", f"{status['metrics']['sarima']['mape']:.2f}%")
                st.json(status)
            else:
                st.error(f"Échec: {status.get('error')}")
        except requests.RequestException as e:
            st.error(f"Impossible de contacter train-api ({TRAIN_API_URL}): {e}")

# ---------------------------------------------------------------------------
elif page == "Prévision":
    st.header("Prévision")
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
        payload = {"horizon": horizon}
        if dju_forecast:
            payload["dju_forecast"] = dju_forecast
        try:
            resp = requests.post(f"{PREDICT_API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 503:
                st.warning("Aucun modèle entraîné — lancez d'abord un entraînement.")
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
else:
    st.header("Données historiques")
    try:
        conso = pd.read_csv("data/raw/consommation_mensuelle.csv", parse_dates=["Mois"])
        dju = pd.read_csv("data/raw/dju_mensuel.csv", parse_dates=["Mois"])
        merged = conso.merge(dju, on="Mois")

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
        st.warning("data/raw/*.csv introuvable dans ce conteneur — montez le dossier data/ (voir docker-compose.yml).")
