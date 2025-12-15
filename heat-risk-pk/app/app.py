import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path
import json
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Urban Heat Stress Risk – Pakistan",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "outputs" / "forecasts"
FIG_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}
RISK_COLORS = {
    0: [0, 200, 0],
    1: [255, 200, 0],
    2: [255, 120, 0],
    3: [200, 0, 0],
}

CITY_COORDS = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5497, 74.3436),
    "Multan": (30.1575, 71.5249),
    "Islamabad": (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_forecast(horizon, scenario_slug):
    fname = f"forecast_{horizon}m_{scenario_slug}.csv"
    path = FORECAST_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing forecast file:\n{path}\n\n"
            f"Expected files like:\n"
            f"- forecast_6m_baseline.csv\n"
            f"- forecast_6m_plus1c.csv\n"
            f"- forecast_6m_plus2c.csv"
        )
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_metrics():
    path = FIG_DIR / "model_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_model_notes():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_history_if_exists():
    path = DATA_PROCESSED / "df_model_forecast.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def apply_whatif_rescore(df, temp_delta, urban_delta, pop_mult):
    """
    Instant re-scoring layer for PoC interactivity.
    Adjusts p_extreme and renormalizes probabilities based on sliders.
    This is a "what-if sensitivity layer" (not retraining).
    """
    df2 = df.copy()

    # sensitivity score (tunable)
    score = (0.35 * temp_delta) + (0.08 * urban_delta) + (0.60 * (pop_mult - 1.0))
    bump = 1 / (1 + np.exp(-score))  # 0..1

    df2["p_extreme_adj"] = np.clip(df2["p_extreme"] + 0.35 * (bump - 0.5), 0, 1)
    take = df2["p_extreme_adj"] - df2["p_extreme"]

    df2["p_low_adj"] = np.clip(df2["p_low"] - 0.70 * take, 0, 1)
    df2["p_mod_adj"] = np.clip(df2["p_mod"] - 0.20 * take, 0, 1)
    df2["p_high_adj"] = np.clip(df2["p_high"] - 0.10 * take, 0, 1)

    s = df2["p_low_adj"] + df2["p_mod_adj"] + df2["p_high_adj"] + df2["p_extreme_adj"]
    for c in ["p_low_adj", "p_mod_adj", "p_high_adj", "p_extreme_adj"]:
        df2[c] = df2[c] / s

    df2["expected_risk_adj"] = (
        0 * df2["p_low_adj"] + 1 * df2["p_mod_adj"] + 2 * df2["p_high_adj"] + 3 * df2["p_extreme_adj"]
    )
    df2["pred_risk_adj"] = df2[["p_low_adj", "p_mod_adj", "p_high_adj", "p_extreme_adj"]].values.argmax(axis=1)
    df2["risk_name_adj"] = df2["pred_risk_adj"].map(RISK_LABELS)
    return df2

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("🔧 Controls")

horizon = st.sidebar.selectbox("Forecast Horizon (months)", [6, 12, 24], index=0)

scenario_label = st.sidebar.selectbox(
    "Climate Scenario",
    ["Baseline", "+1°C warming", "+2°C warming"],
    index=0
)
scenario_map = {"Baseline": "baseline", "+1°C warming": "plus1c", "+2°C warming": "plus2c"}
scenario = scenario_map[scenario_label]

selected_city = st.sidebar.selectbox(
    "City Drill-Down",
    ["All Cities", "Karachi", "Lahore", "Multan", "Islamabad", "Rawalpindi", "Peshawar", "Quetta"]
)

compare_cities = st.sidebar.multiselect(
    "Compare Cities (multi-line chart)",
    ["Karachi", "Lahore", "Multan", "Islamabad", "Rawalpindi", "Peshawar", "Quetta"],
    default=["Karachi", "Lahore", "Multan"]
)

st.sidebar.subheader("🧪 What-if Simulation (Instant)")
temp_delta = st.sidebar.slider("Temperature delta (°C)", -1.0, 4.0, 0.0, 0.5)
urban_delta = st.sidebar.slider("Urbanization delta (pp)", -5.0, 10.0, 0.0, 0.5)
pop_mult = st.sidebar.slider("Population multiplier", 0.8, 1.5, 1.0, 0.05)
predict_now = st.sidebar.button("🚀 Predict Now", type="primary")

# -----------------------------
# Load forecast data
# -----------------------------
try:
    df = load_forecast(horizon, scenario)
except Exception as e:
    st.error(str(e))
    st.stop()

# Date + label setup
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

# base outputs
df["risk_name"] = df["pred_risk"].map(RISK_LABELS)
df["expected_risk"] = (0 * df.p_low + 1 * df.p_mod + 2 * df.p_high + 3 * df.p_extreme)

# Display window (so you can show 2026 if present)
st.sidebar.subheader("📅 Display Window")
st.sidebar.caption(f"Forecast years available: {int(df['year'].min())} → {int(df['year'].max())}")
year_from = st.sidebar.number_input("Show from year", value=int(df["year"].min()), step=1)
year_to = st.sidebar.number_input("Show until year", value=int(df["year"].max()), step=1)
df = df[(df["year"] >= year_from) & (df["year"] <= year_to)].copy()

# What-if view state (persistent)
if "use_whatif" not in st.session_state:
    st.session_state["use_whatif"] = False

if predict_now:
    st.session_state["use_whatif"] = True
    st.success("✅ What-if prediction updated instantly.")

df_view = df.copy()

# Apply what-if rescore if enabled
if st.session_state["use_whatif"]:
    df_view = apply_whatif_rescore(df_view, temp_delta, urban_delta, pop_mult)
    df_view["pred_risk_view"] = df_view["pred_risk_adj"]
    df_view["risk_name_view"] = df_view["risk_name_adj"]
    df_view["expected_risk_view"] = df_view["expected_risk_adj"]
    df_view["p_extreme_view"] = df_view["p_extreme_adj"]
else:
    df_view["pred_risk_view"] = df_view["pred_risk"]
    df_view["risk_name_view"] = df_view["risk_name"]
    df_view["expected_risk_view"] = df_view["expected_risk"]
    df_view["p_extreme_view"] = df_view["p_extreme"]

# -----------------------------
# HEADER
# -----------------------------
st.title("🔥 Urban Heat Stress Risk Forecasting – Pakistan")
st.markdown("""
**End-to-end ML decision-support system** for forecasting urban heat stress risk  
Includes: model comparison, explainability, interactive what-if simulation, and city risk monitoring.
""")

# -----------------------------
# KPI ROW
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cities Analysed", df_view["city"].nunique())
k2.metric("Forecast Horizon", f"{horizon} months")
k3.metric("Cities w/ any Extreme month", int((df_view["pred_risk_view"] == 3).groupby(df_view["city"]).any().sum()))
k4.metric("Max P(Extreme)", f"{df_view['p_extreme_view'].max():.2f}")

st.divider()

# =====================================================
# SECTION 1 — MAP
# =====================================================
st.header("🗺 Pakistan Urban Heat Stress Risk Map (Forecast)")

map_df = (
    df_view.groupby("city")
    .agg(
        avg_risk=("expected_risk_view", "mean"),
        max_extreme=("p_extreme_view", "max"),
        risk=("pred_risk_view", "max"),
    )
    .reset_index()
)

map_df["lat"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
map_df["lon"] = map_df["city"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])
map_df = map_df.dropna(subset=["lat", "lon"])
map_df["color"] = map_df["risk"].map(RISK_COLORS)

layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position=["lon", "lat"],
    get_radius="max_extreme * 50000 + 15000",
    get_fill_color="color",
    pickable=True,
)

view_state = pdk.ViewState(latitude=30.5, longitude=69.0, zoom=5, pitch=0)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{city}\nAvg risk: {avg_risk}\nMax P(extreme): {max_extreme}"}
    )
)

st.divider()

# =====================================================
# SECTION 2 — TOP CITIES TABLE + INSIGHTS
# =====================================================
st.header("🏙 Top Cities at Risk (Ranked)")

top = (
    df_view.groupby("city")
    .agg(
        avg_expected_risk=("expected_risk_view", "mean"),
        max_extreme_prob=("p_extreme_view", "max"),
        extreme_months=("pred_risk_view", lambda s: int((s == 3).sum())),
    )
    .sort_values("avg_expected_risk", ascending=False)
)

st.dataframe(top.style.background_gradient(cmap="Reds"), use_container_width=True)

# Quick insights
worst = top.index[0]
st.markdown("### 🔎 Key Insights (Auto)")
st.write(f"- **Highest average expected risk:** {worst} (avg={top.iloc[0]['avg_expected_risk']:.2f})")
st.write(f"- **Highest max extreme probability:** {top['max_extreme_prob'].idxmax()} (max={top['max_extreme_prob'].max():.2f})")
st.write(f"- **Total Extreme months across all cities:** {int((df_view['pred_risk_view']==3).sum())}")

st.divider()

# =====================================================
# SECTION 3 — CITY TIMELINE + COMPARISON
# =====================================================
st.header("📈 City Risk Timeline")

if selected_city != "All Cities":
    city_df = df_view[df_view["city"] == selected_city].sort_values("date").copy()

    fig = px.line(
        city_df,
        x="date",
        y="expected_risk_view",
        title=f"Expected Heat Stress Risk – {selected_city}",
        markers=True,
        hover_data={
            "year": True,
            "month": True,
            "risk_name_view": True,
            "p_extreme_view": ":.3f",
        }
    )
    fig.update_yaxes(title="Expected Risk (0=Low → 3=Extreme)", range=[0, 3])
    fig.update_xaxes(title="Month")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monthly forecast table**")
    st.dataframe(city_df[["year", "month", "risk_name_view", "p_extreme_view", "expected_risk_view"]], use_container_width=True)
else:
    st.info("Select a city from the sidebar to view its timeline.")

st.subheader("📊 City Comparison (Selected Cities)")

comp_df = df_view[df_view["city"].isin(compare_cities)].sort_values("date").copy()
if len(comp_df):
    fig2 = px.line(
        comp_df,
        x="date",
        y="expected_risk_view",
        color="city",
        title="Expected Risk Comparison Across Cities"
    )
    fig2.update_yaxes(title="Expected Risk (0..3)", range=[0, 3])
    fig2.update_xaxes(title="Month")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        comp_df,
        x="date",
        y="p_extreme_view",
        color="city",
        title="Extreme Probability Comparison Across Cities"
    )
    fig3.update_yaxes(title="P(Extreme)")
    fig3.update_xaxes(title="Month")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No cities selected for comparison.")

st.divider()

# =====================================================
# SECTION 4 — PAST RECORDS DRILL-DOWN (if processed history exists)
# =====================================================
st.header("🕰 Past Records (City → Month → Details)")

hist = load_history_if_exists()
if hist is None:
    st.info("Historical processed file not found: data/processed/df_model_forecast.csv (optional section).")
else:
    hist["date"] = pd.to_datetime(hist["year"].astype(str) + "-" + hist["month"].astype(str).str.zfill(2) + "-01")
    hist_cities = sorted(hist["city"].unique().tolist())

    c_left, c_right = st.columns([1, 2])

    with c_left:
        h_city = st.selectbox("City (history)", hist_cities)
        h_df = hist[hist["city"] == h_city].sort_values("date").copy()
        h_df["record_key"] = h_df["year"].astype(str) + "-" + h_df["month"].astype(str).str.zfill(2)
        rec_key = st.selectbox("Pick a historical record (year-month)", h_df["record_key"].tolist(), index=len(h_df)-1)
        rec = h_df[h_df["record_key"] == rec_key].iloc[0]

        st.markdown("### Record Details")
        st.write(f"**Record:** {rec_key}")
        if "risk_label" in h_df.columns:
            st.write(f"**Observed Risk:** {int(rec['risk_label'])} ({RISK_LABELS.get(int(rec['risk_label']), 'NA')})")
        for col in ["tavg_mean", "tmax_mean", "heat_stress_index", "pop_density", "urban_pct"]:
            if col in h_df.columns:
                val = rec.get(col, np.nan)
                st.write(f"- **{col}:** {val:.3f}" if pd.notna(val) else f"- **{col}:** NA")

    with c_right:
        st.markdown("### Historical Trends")
        ycol = "heat_stress_index" if "heat_stress_index" in h_df.columns else None
        if ycol:
            st.plotly_chart(px.line(h_df, x="date", y=ycol, title=f"{h_city}: Heat Stress Index (History)"), use_container_width=True)
        if "tmax_mean" in h_df.columns:
            st.plotly_chart(px.line(h_df, x="date", y="tmax_mean", title=f"{h_city}: Tmax Mean (History)"), use_container_width=True)

st.divider()

# =====================================================
# SECTION 5 — MODEL PERFORMANCE
# =====================================================
st.header("🧠 Model Performance & Selection")

metrics = load_metrics()
if metrics is None:
    st.warning("model_metrics.csv not found. Run: python -m src.evaluate_all_models")
else:
    st.dataframe(metrics.style.highlight_max(axis=0), use_container_width=True)

st.markdown("""
**Model Selection Criteria**
- Primary: **Macro-F1** (handles class imbalance)
- Safety-critical: **Extreme-class Recall**
- Selected model: **Climate-only Gradient Boosting**
""")

cm_cols = st.columns(3)
for img in ["confusion_matrix_baseline.png", "confusion_matrix_logreg.png", "confusion_matrix_hgb.png"]:
    p = FIG_DIR / img
    if p.exists():
        cm_cols.pop(0).image(str(p), caption=img, use_container_width=True)

st.divider()

# =====================================================
# SECTION 6 — FEATURE IMPORTANCE + SHAP (if figures exist)
# =====================================================
st.header("🔍 Feature Contribution & Explainability")

c1, c2 = st.columns(2)
p1 = FIG_DIR / "perm_importance_forecast_hgb_top15.png"
p2 = FIG_DIR / "rf_feature_importance_top15.png"
if p1.exists():
    c1.image(str(p1), caption="Permutation Importance (Deployed Forecast Model)", use_container_width=True)
else:
    c1.info("Run: python -m src.feature_importance")

if p2.exists():
    c2.image(str(p2), caption="Random Forest Feature Importance", use_container_width=True)

st.subheader("🧩 SHAP Explainability")
c3, c4 = st.columns(2)
s1 = FIG_DIR / "shap_summary_extreme.png"
s2 = FIG_DIR / "shap_waterfall_example.png"
if s1.exists():
    c3.image(str(s1), caption="Global SHAP Summary – Extreme Risk", use_container_width=True)
else:
    c3.info("Run: python -m src.explain")

if s2.exists():
    c4.image(str(s2), caption="Local SHAP Waterfall – Example", use_container_width=True)

st.divider()

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
---
### 📌 Project Summary
- **Task:** Urban Heat Stress Risk Forecasting (Pakistan cities)
- **Data:** 4 public heterogeneous sources (weather, population density, urban %, surface temperature)
- **ML:** Baseline + classical models + ensemble + explainability
- **PoC:** Interactive dashboard, map, what-if simulation, historical drill-down
""")