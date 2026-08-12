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
# de JWT"). Interface utilisateur en anglais ; commentaires de
# code en francais, comme le reste du depot.
#
# Pages :
#   - "Training" : bouton POST /train {country}, affichage du statut/MAPE
#   - "Forecast" : formulaire horizon (+ DJU previsionnel optionnel),
#     graphique historique + prevision + intervalle de confiance
#   - "Historical Data" : courbe Consommation / DJU brute (pays selectionne)
#   - "2019 Comparison" : tous les pays cote a cote sur leur seule annee civile
#     commune complete — conso totale, conso/habitant, profil saisonnier,
#     sensibilite a la temperature (pente + R² de Consommation ~ DJU)
#
# Le pays selectionne dans la barre laterale est garde dans
# st.session_state["country"] et reutilise sur les pages "Training" /
# "Forecast" / "Historical Data". "2019 Comparison" ignore ce choix —
# elle affiche systematiquement tous les pays disponibles.
#
# Dependances externes : streamlit, requests, pandas, numpy, plotly
# ============================================================
import math
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy import stats as scipy_stats
import streamlit as st

TRAIN_API_URL = os.getenv("TRAIN_API_URL", "http://localhost:5010")
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:5011")

st.set_page_config(page_title="elec_prediction", layout="wide")
st.title("⚡ Electricity Consumption Forecasting")
st.caption("Temperature correction (DJU) + Holt-Winters / SARIMA(0,1,1)(1,1,1)₁₂")


@st.cache_data(ttl=60)
def _fetch_countries():
    resp = requests.get(f"{TRAIN_API_URL}/countries", timeout=10)
    resp.raise_for_status()
    return resp.json()


# Flag emoji per country code — falls back to no flag for a country added later without
# an entry here, same defensive spirit as the "Comparaison"/color-fallback pattern below.
COUNTRY_FLAGS = {
    "france": "🇫🇷", "usa": "🇺🇸", "germany": "🇩🇪", "uk": "🇬🇧", "finland": "🇫🇮",
    "austria": "🇦🇹", "luxembourg": "🇱🇺",
}

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
    "Country",
    options=list(_country_options.keys()),
    format_func=lambda code: f"{COUNTRY_FLAGS.get(code, '')} {_country_options.get(code, code)}".strip(),
    key="country",
)

page = st.sidebar.radio("Page", ["Training", "Forecast", "Historical Data", "2019 Comparison"])

# ---------------------------------------------------------------------------
if page == "Training":
    st.header(f"Model Training — {_country_options.get(country, country)}")
    st.write("Runs the pipeline: OLS regression (Consommation ~ DJU) → deseasonalization → "
             "Holt-Winters + SARIMA → backtest on the test year → best-model selection.")

    if st.button("Start training", type="primary"):
        try:
            resp = requests.post(f"{TRAIN_API_URL}/train", json={"country": country}, timeout=10)
            if resp.status_code == 400:
                st.error(f"Invalid country: {resp.json().get('detail')}")
            else:
                resp.raise_for_status()
                job = resp.json()
                job_id = job["job_id"]
                st.info(f"Job started: `{job_id}`")

                progress = st.empty()
                with st.spinner("Training in progress..."):
                    for _ in range(120):  # up to ~2 min polling
                        time.sleep(1)
                        status = requests.get(f"{TRAIN_API_URL}/train/status/{job_id}", timeout=10).json()
                        if status["status"] in ("success", "failed"):
                            break
                        progress.text(f"status: {status['status']}...")

                progress.empty()

                if status["status"] == "success":
                    st.success(f"Model selected: **{status['best_model']}**")
                    model_labels = {"holt_winters": "Holt-Winters", "sarima": "SARIMA", "ml_global": "LightGBM (global)"}
                    result_metrics = status["metrics"]
                    cols = st.columns(len(result_metrics))
                    for col, (name, m) in zip(cols, result_metrics.items()):
                        col.metric(f"{model_labels.get(name, name)} MAPE", f"{m['mape']:.2f}%")
                    st.json(status)

                    try:
                        reload_resp = requests.post(
                            f"{PREDICT_API_URL}/reload-artifacts", params={"country": country}, timeout=10,
                        )
                        reload_resp.raise_for_status()
                        st.info("predict-api reloaded with the new model.")
                    except requests.RequestException as e:
                        st.warning(f"Training succeeded but couldn't reload predict-api ({PREDICT_API_URL}): {e}")
                else:
                    st.error(f"Failed: {status.get('error')}")
        except requests.RequestException as e:
            st.error(f"Could not reach train-api ({TRAIN_API_URL}): {e}")

# ---------------------------------------------------------------------------
elif page == "Forecast":
    st.header(f"Forecast — {_country_options.get(country, country)}")
    horizon = st.slider("Horizon (months)", min_value=1, max_value=36, value=12)
    use_dju = st.checkbox("Provide a forecast DJU (to reconstruct actual consumption)")
    dju_forecast = None
    if use_dju:
        dju_text = st.text_input(f"Forecast DJU ({horizon} comma-separated values)")
        if dju_text:
            try:
                dju_forecast = [float(x.strip()) for x in dju_text.split(",")]
            except ValueError:
                st.warning("Invalid format — use comma-separated numbers.")

    if st.button("Forecast", type="primary"):
        payload = {"horizon": horizon, "country": country}
        if dju_forecast:
            payload["dju_forecast"] = dju_forecast
        try:
            resp = requests.post(f"{PREDICT_API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 503:
                st.warning(f"No trained model for {_country_options.get(country, country)} — run a training job first.")
            else:
                resp.raise_for_status()
                result = resp.json()
                st.write(f"Model used: **{result['model_used']}** · Last observed month: {result['last_observed_month']}")

                df = pd.DataFrame(result["forecast"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["month"], y=df["conso_correction_pred_gwh"], name="Corrected consumption (forecast)", mode="lines+markers"))
                if "ci_lower_gwh" in df.columns:
                    fig.add_trace(go.Scatter(x=df["month"], y=df["ci_upper_gwh"], line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=df["month"], y=df["ci_lower_gwh"], fill="tonexty", line=dict(width=0), name="95% interval"))
                if "consommation_pred_gwh" in df.columns:
                    fig.add_trace(go.Scatter(x=df["month"], y=df["consommation_pred_gwh"], name="Reconstructed actual consumption", mode="lines+markers"))
                fig.update_layout(xaxis_title="Month", yaxis_title="GWh")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True)
        except requests.RequestException as e:
            st.error(f"Could not reach predict-api ({PREDICT_API_URL}): {e}")

# ---------------------------------------------------------------------------
elif page == "Historical Data":
    st.header(f"Historical Data — {_country_options.get(country, country)}")
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
                "Date range", min_value=min_date, max_value=max_date,
                value=(min_date, max_date), format="YYYY-MM", key=f"date_range_{country}",
            )
            merged = merged[(merged["Mois"].dt.date >= date_range[0]) & (merged["Mois"].dt.date <= date_range[1])]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged["Mois"], y=merged["Consommation"], name="Consumption (GWh)"))
        fig.add_trace(go.Scatter(x=merged["Mois"], y=merged["DJU"], name="DJU", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Consumption (GWh)"),
            yaxis2=dict(title="DJU", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged, use_container_width=True)
    except FileNotFoundError:
        st.warning(f"data/raw/{country}/*.csv not found — this country hasn't been synced yet "
                   f"(see scripts/sync_country_data.py) or the data/ folder isn't mounted (see docker-compose.yml).")

# ---------------------------------------------------------------------------
else:  # page == "2019 Comparison"
    st.header("2019 Comparison — All Countries")
    st.write(
        "2019 is the only calendar year every country has a complete series for "
        "(France: 2009-2019 · the others: 2015/2019-2024) — so it's the only basis "
        "for a direct country-to-country comparison."
    )

    # Fixed hue per country (never remapped) — same order as connectors/registry.py.
    # A country added later without a slot here falls back to neutral gray rather than crash.
    COUNTRY_COLORS = {
        "france": "#2a78d6",     # blue
        "usa": "#eb6834",        # orange
        "germany": "#1baf7a",    # aqua
        "uk": "#eda100",         # yellow
        "finland": "#e87ba4",    # magenta
        "austria": "#008300",    # green
        "luxembourg": "#4a3aa7",  # violet
    }
    GRID_COLOR = "#e1e0d9"
    FALLBACK_COLOR = "#898781"

    # World Bank 2019 estimates (the only external figures on this page — everything else
    # is computed from data/raw/*). Population in millions; GDP in billion current USD.
    POPULATION_M = {
        "france": 67.4, "germany": 83.1, "usa": 328.2, "uk": 66.8, "finland": 5.52,
        "austria": 8.86, "luxembourg": 0.614,
    }
    GDP_BILLION_USD = {
        "france": 2729, "germany": 3889, "usa": 21433, "uk": 2851, "finland": 269,
        "austria": 446, "luxembourg": 71,
    }
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def _ols_with_inference(x: np.ndarray, y: np.ndarray):
        """Simple bivariate OLS y ~ x with closed-form standard error / t-test / 95% CI on the
        slope (textbook formula — the same result statsmodels.OLS would give, without adding
        that dependency here). n=12 (one calendar year) is a real statistical-power limitation:
        see the Methodology note below — this is exactly why the CI/p-value are surfaced at all
        rather than just the point estimate."""
        n = len(x)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        dof = n - 2
        mse = np.sum(resid ** 2) / dof
        se_slope = np.sqrt(mse / np.sum((x - x.mean()) ** 2))
        t_stat = slope / se_slope
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), dof))
        ci_halfwidth = scipy_stats.t.ppf(0.975, dof) * se_slope
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        return {
            "slope": slope, "intercept": intercept, "r2": r2, "se": se_slope,
            "p_value": p_value, "ci_low": slope - ci_halfwidth, "ci_high": slope + ci_halfwidth,
        }

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
            if len(df) < 12 or c not in POPULATION_M or c not in GDP_BILLION_USD:
                missing.append(c)
                continue

            total = df["Consommation"].sum()
            pop = POPULATION_M[c]
            gdp = GDP_BILLION_USD[c]
            per_capita_mwh = (total * 1000) / (pop * 1e6)  # GWh -> MWh, then / persons
            energy_intensity = total / gdp  # GWh per billion USD == MWh per million USD GDP
            peak_idx, trough_idx = df["Consommation"].idxmax(), df["Consommation"].idxmin()
            peak_trough_ratio = df["Consommation"].max() / df["Consommation"].min()

            ols = _ols_with_inference(df["DJU"].values, df["Consommation"].values)

            profile_pct = (df["Consommation"] / total * 100).values
            profiles[c] = profile_pct
            raw[c] = df

            rows.append({
                "code": c,
                "total_gwh": total,
                "population_m": pop,
                "per_capita_mwh": per_capita_mwh,
                "energy_intensity": energy_intensity,
                "peak_month": df.loc[peak_idx, "Mois"].strftime("%b"),
                "trough_month": df.loc[trough_idx, "Mois"].strftime("%b"),
                "peak_trough_ratio": peak_trough_ratio,
                "dju_slope": ols["slope"],
                "dju_slope_se": ols["se"],
                "dju_slope_ci_low": ols["ci_low"],
                "dju_slope_ci_high": ols["ci_high"],
                "dju_slope_pvalue": ols["p_value"],
                "dju_r2": ols["r2"],
                "winter_share_pct": profile_pct[[0, 1, 11]].sum(),
                "summer_share_pct": profile_pct[[5, 6, 7]].sum(),
            })
        return pd.DataFrame(rows), profiles, raw, missing

    summary, profiles, raw_2019, missing = _load_2019_comparison(tuple(sorted(_country_options.keys())))

    if missing:
        st.info(
            "Not included (incomplete 2019 data or country not yet synced): "
            + ", ".join(_country_options.get(c, c) for c in missing)
            + " — see `scripts/sync_country_data.py`."
        )

    if summary.empty:
        st.error("No country has a complete 2019 year yet.")
    else:
        summary["Country"] = summary["code"].map(lambda c: _country_options.get(c, c))
        colors = [COUNTRY_COLORS.get(c, FALLBACK_COLOR) for c in summary["code"]]

        # --- Headline KPIs ---
        top_capita = summary.loc[summary["per_capita_mwh"].idxmax()]
        low_capita = summary.loc[summary["per_capita_mwh"].idxmin()]
        top_intensity = summary.loc[summary["energy_intensity"].idxmax()]
        top_r2 = summary.loc[summary["dju_r2"].idxmax()]
        summer_peaking = summary[summary["summer_share_pct"] > summary["winter_share_pct"]]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Highest per-capita consumption",
            top_capita["Country"],
            f"{top_capita['per_capita_mwh']:.1f} MWh/capita — {top_capita['per_capita_mwh'] / low_capita['per_capita_mwh']:.1f}× {low_capita['Country']}",
        )
        k2.metric(
            "Most electricity-intensive economy",
            top_intensity["Country"],
            f"{top_intensity['energy_intensity']:.1f} MWh / M$ GDP",
        )
        k3.metric(
            "Most temperature-sensitive",
            top_r2["Country"],
            f"R² = {top_r2['dju_r2']:.2f} (p = {top_r2['dju_slope_pvalue']:.3f})",
        )
        if not summer_peaking.empty:
            k4.metric("Only summer-peaking country(ies)", ", ".join(summer_peaking["Country"]), "AC > heating")
        else:
            k4.metric("Seasonal profile", "All winter-peaking", "heating > AC, everywhere")

        with st.expander("Methodology & limitations — read before over-interpreting the R² values"):
            st.markdown(
                "- **Per-capita vs. energy intensity** answer different questions: per-capita "
                "(MWh/person) is a consumption-volume/lifestyle measure; energy intensity "
                "(MWh per million USD of GDP, using 2019 World Bank GDP) is closer to an "
                "economic-efficiency measure, since it controls for wealth and economic "
                "structure. Reporting only per-capita — as many quick country comparisons do — "
                "conflates the two; both are shown here for that reason.\n"
                "- **Heating-only degree-days structurally under-explain AC-heavy countries.** "
                "The DJU regression here only models heating demand (`DJU = max(18°C − T, 0)`), "
                "with no cooling term. For the USA — the one country in this set with material "
                "summer air-conditioning load — electricity demand is well documented in the "
                "literature to be U-shaped in temperature (Deschênes & Greenstone), rising at "
                "*both* cold and hot extremes. A heating-only linear model only captures the cold "
                "arm of that curve, so USA's low R² here is a **specification artifact of a "
                "heating-only model**, not evidence that US demand is temperature-insensitive — "
                "a combined HDD+CDD model would very likely fit it far better.\n"
                "- **n = 12 per country is a real statistical-power limit.** Each regression uses "
                "one calendar year (12 monthly points, 10 degrees of freedom for a simple "
                "bivariate OLS). M&V practice (ASHRAE Guideline 14) treats a single year as the "
                "bare minimum usable sample, not a robust one — one unusual month can swing the "
                "slope and R² noticeably. The 95% confidence interval and p-value on the slope "
                "(table below) exist specifically so this isn't read as more precise than it is; "
                "a multi-year pooled regression would be the standard fix.\n"
                "- **The 18°C base temperature is applied uniformly to all countries on purpose.** "
                "Real national conventions differ (US utilities use 65°F/18.3°C, UK historically "
                "15.5°C, Germany ~15°C) — but those are institutional conventions, not physical "
                "constants. For a *cross-country* comparison specifically, one consistent base "
                "removes that confound so slope/R² differences reflect actual demand behavior, "
                "not differing degree-day definitions.\n"
                "- **2019 is a single-year snapshot**, not a multi-year trend — chosen because "
                "it's the one calendar year every country's data overlaps (see the page intro). "
                "A single year can be pulled around by one unusually warm or cold month; treat "
                "findings here as indicative, not as a robust multi-year climate-demand estimate."
            )

        st.markdown("---")

        # --- Total consumption (raw scale) ---
        st.subheader("Total 2019 Consumption")
        fig1 = go.Figure(go.Bar(
            x=summary["Country"], y=summary["total_gwh"], marker_color=colors,
            text=summary["total_gwh"].apply(lambda v: f"{v / 1000:,.0f} TWh"), textposition="outside",
        ))
        fig1.update_layout(yaxis_title="GWh", showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Raw scale — mostly reflects country size, not per-capita consumption or "
                   "economic efficiency (see next two charts).")

        # --- Per-capita consumption ---
        st.subheader("2019 Per-Capita Consumption")
        fig2 = go.Figure(go.Bar(
            x=summary["Country"], y=summary["per_capita_mwh"], marker_color=colors,
            text=summary["per_capita_mwh"].apply(lambda v: f"{v:.1f}"), textposition="outside",
        ))
        fig2.update_layout(yaxis_title="MWh / capita", showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("A consumption-volume/lifestyle measure. 2019 population (World Bank, rounded): " + ", ".join(
            f"{_country_options.get(c, c)} {POPULATION_M[c]:.1f}M" for c in summary["code"]
        ))

        # --- Energy intensity (GDP-normalized) ---
        st.subheader("2019 Energy Intensity — Electricity per Unit GDP")
        fig_intensity = go.Figure(go.Bar(
            x=summary["Country"], y=summary["energy_intensity"], marker_color=colors,
            text=summary["energy_intensity"].apply(lambda v: f"{v:.1f}"), textposition="outside",
        ))
        fig_intensity.update_layout(yaxis_title="MWh / million USD GDP", showlegend=False, yaxis=dict(gridcolor=GRID_COLOR))
        st.plotly_chart(fig_intensity, use_container_width=True)
        st.caption(
            "An economic-efficiency measure, not a lifestyle one — controls for wealth and "
            "economic structure, unlike per-capita above. 2019 GDP, current USD billions "
            "(World Bank, rounded): " + ", ".join(
                f"{_country_options.get(c, c)} ${GDP_BILLION_USD[c]:,.0f}B" for c in summary["code"]
            )
        )

        # --- Seasonality profile ---
        st.subheader("Seasonality Profile — Share of Each Month in the Annual Total")
        fig3 = go.Figure()
        for c in summary["code"]:
            fig3.add_trace(go.Scatter(
                x=MONTHS, y=profiles[c], name=_country_options.get(c, c), mode="lines+markers",
                line=dict(color=COUNTRY_COLORS.get(c, FALLBACK_COLOR), width=2), marker=dict(size=8),
            ))
        fig3.update_layout(
            yaxis_title="% of annual total", yaxis=dict(gridcolor=GRID_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, use_container_width=True)
        if not summer_peaking.empty:
            st.info(
                f"**{', '.join(summer_peaking['Country'])}** is the only profile where summer (Jun+Jul+Aug, "
                f"{summer_peaking.iloc[0]['summer_share_pct']:.1f}% of the total) outweighs winter "
                f"(Dec+Jan+Feb, {summer_peaking.iloc[0]['winter_share_pct']:.1f}%) — a typical signature of "
                f"air-conditioning-driven demand rather than electric heating."
            )

        # --- Cross-country correlation of seasonal shape (which countries behave alike?) ---
        st.subheader("Seasonal-Profile Correlation Between Countries")
        codes_corr = list(summary["code"])
        profile_matrix = np.array([profiles[c] for c in codes_corr])  # rows=countries, cols=months
        corr_matrix = np.corrcoef(profile_matrix)
        country_labels = [_country_options.get(c, c) for c in codes_corr]
        # Diverging blue<->red, gray midpoint at 0 — correlation has a meaningful zero (no
        # relationship), so this is a polarity encoding, not a plain magnitude/sequential one.
        diverging_scale = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix, x=country_labels, y=country_labels, zmin=-1, zmax=1,
            colorscale=diverging_scale, colorbar=dict(title="r"),
            text=np.round(corr_matrix, 2), texttemplate="%{text}",
        ))
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "Pearson correlation between each pair of countries' monthly seasonality profiles "
            "(the % curves above) — high positive r means two countries share the same "
            "month-to-month shape (typically the heating-dominated European countries with each "
            "other), not that their consumption *levels* are similar. This is a shape comparison, "
            "computed independently of the DJU regression above."
        )

        # --- Temperature sensitivity: small multiples (one panel per country, own scale) ---
        st.subheader("Temperature Sensitivity (Consumption vs DJU, 2019)")
        codes = list(summary["code"])
        n_cols = min(4, len(codes))  # wrap into a grid instead of one ever-widening row
        n_rows = math.ceil(len(codes) / n_cols)
        fig4 = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[
            f"{_country_options.get(c, c)} (R²={summary.set_index('code').loc[c, 'dju_r2']:.2f})" for c in codes
        ])
        for i, c in enumerate(codes):
            row, col = (i // n_cols) + 1, (i % n_cols) + 1
            df = raw_2019[c]
            color = COUNTRY_COLORS.get(c, FALLBACK_COLOR)
            fig4.add_trace(go.Scatter(x=df["DJU"], y=df["Consommation"], mode="markers",
                                       marker=dict(color=color, size=8), showlegend=False), row=row, col=col)
            slope, intercept = np.polyfit(df["DJU"], df["Consommation"], 1)
            x_line = np.array([df["DJU"].min(), df["DJU"].max()])
            fig4.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                       line=dict(color=color, width=2, dash="dot"), showlegend=False), row=row, col=col)
            fig4.update_xaxes(title_text="DJU", row=row, col=col, gridcolor=GRID_COLOR)
            fig4.update_yaxes(title_text="GWh" if col == 1 else None, row=row, col=col, gridcolor=GRID_COLOR)
        fig4.update_layout(height=350 * n_rows)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "Each panel has its own scale — the point is to compare slope and spread, not absolute "
            "levels. R² close to 1: heating-degree-days alone explain nearly all the monthly "
            "variation. Low R² can mean weaker temperature sensitivity — **or** a country with "
            "material summer cooling load, which this heating-only model has no term for at all "
            "(see the Methodology note above, and the USA panel in particular)."
        )

        # --- Summary table (accessible fallback for every chart above) ---
        st.subheader("Summary Table")
        summary["dju_slope_ci"] = summary.apply(
            lambda r: f"[{r['dju_slope_ci_low']:.1f}, {r['dju_slope_ci_high']:.1f}]", axis=1
        )
        display_cols = {
            "Country": "Country", "total_gwh": "Total (GWh)", "per_capita_mwh": "Per Capita (MWh)",
            "energy_intensity": "Energy Intensity (MWh/M$)",
            "peak_month": "Peak Month", "trough_month": "Trough Month",
            "peak_trough_ratio": "Peak/Trough Ratio", "dju_r2": "DJU Sensitivity (R²)",
            "dju_slope_pvalue": "DJU Slope p-value", "dju_slope_ci": "DJU Slope 95% CI",
            "winter_share_pct": "Winter Share %", "summer_share_pct": "Summer Share %",
        }
        table = summary[list(display_cols.keys())].rename(columns=display_cols)
        numeric_cols = table.select_dtypes(include="number").columns
        table[numeric_cols] = table[numeric_cols].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "DJU Slope 95% CI / p-value: inference on the heating-degree-day regression slope "
            "(n=12 per country — see Methodology note above). A p-value above 0.05 means the "
            "slope isn't statistically distinguishable from zero at that country's data volume."
        )
