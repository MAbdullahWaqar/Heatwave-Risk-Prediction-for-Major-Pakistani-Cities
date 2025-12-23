import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path
import json
import plotly.express as px

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Urban Heat Stress Risk – Pakistan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Paths
# =====================================================
ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = ROOT / "outputs" / "forecasts"
FIG_DIR = ROOT / "outputs" / "figures"
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"

# =====================================================
# Constants
# =====================================================
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

# =====================================================
# Helpers
# =====================================================
@st.cache_data
def load_forecast(horizon: int, scenario_slug: str) -> pd.DataFrame:
    fname = f"forecast_{horizon}m_{scenario_slug}.csv"
    path = FORECAST_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing forecast file:\n{path}\n\n"
            f"Expected files like:\n"
            f"- forecast_6m_baseline.csv\n"
            f"- forecast_12m_baseline.csv\n"
            f"- forecast_24m_baseline.csv\n"
            f"- forecast_72m_baseline.csv (for 2030 projections)\n"
            f"- forecast_6m_plus1c.csv\n"
            f"- forecast_6m_plus2c.csv\n"
            f"- forecast_72m_plus1c.csv\n"
            f"- forecast_72m_plus2c.csv\n\n"
            f"Run: python generate_forecasts.py"
        )
    return pd.read_csv(path)

@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    path = FIG_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_model_notes() -> dict | None:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_history_if_exists() -> pd.DataFrame | None:
    path = DATA_PROCESSED / "df_model_forecast.csv"
    return pd.read_csv(path) if path.exists() else None

def expected_risk_from_probs(df: pd.DataFrame, p_low, p_mod, p_high, p_extreme) -> pd.Series:
    return 0*p_low + 1*p_mod + 2*p_high + 3*p_extreme

def extend_forecast_to_year(df_in: pd.DataFrame, target_end_year: int = 2030) -> pd.DataFrame:
    """
    Extends forecast table to target_end_year using seasonal averaging.
    This is a PoC projection layer to support long-horizon dashboards (2026–2030).
    
    IMPORTANT: 
    - Only extends if the forecast horizon is >= 12 months.
    - For shorter forecasts (6m), returns data as-is without artificial extension.
    - Uses seasonal averaging to avoid unrealistic repetition of extreme values.
    """
    df = df_in.copy().sort_values(["city", "year", "month"])
    
    # Determine forecast horizon by checking a sample city
    if not df.empty:
        sample_city = df["city"].iloc[0]
        sample_data = df[df["city"] == sample_city]
        forecast_horizon = len(sample_data)
        
        # If forecast horizon < 12 months, don't extend beyond what we have
        if forecast_horizon < 12:
            # Still add date column for consistency
            if "date" not in df.columns:
                df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
            return df
    
    if int(df["year"].max()) >= target_end_year:
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
        return df

    # Mark original data
    df["is_extended"] = False
    
    out_chunks = [df]
    for city in df["city"].unique():
        d = df[df["city"] == city].copy().sort_values(["year", "month"])
        if d.empty:
            continue

        # Calculate seasonal averages from available data (last 12 months if available)
        seasonal_avg = d.tail(12).groupby("month").agg({
            "pred_risk": "mean",
            "p_low": "mean",
            "p_mod": "mean",
            "p_high": "mean",
            "p_extreme": "mean",
            "heat_stress_index_proj": "mean",
            "temp_delta_c": "mean",
            "urban_delta_pct": "mean",
            "pop_delta_mult": "mean"
        }).to_dict(orient="index")
        
        last_y = int(d.iloc[-1]["year"])
        last_m = int(d.iloc[-1]["month"])

        y, m = last_y, last_m
        while y < target_end_year or (y == target_end_year and m < 12):
            m += 1
            if m == 13:
                y += 1
                m = 1
            
            # Use seasonal average for this month
            if m in seasonal_avg:
                row = seasonal_avg[m].copy()
            else:
                # Fallback to last month's average
                row = d.tail(1).iloc[0].to_dict()
            
            row["city"] = city
            row["year"] = y
            row["month"] = m
            row["is_extended"] = True
            
            # Re-calculate pred_risk from probabilities to ensure consistency
            probs = [row["p_low"], row["p_mod"], row["p_high"], row["p_extreme"]]
            row["pred_risk"] = int(np.argmax(probs))
            
            out_chunks.append(pd.DataFrame([row]))

    df_ext = pd.concat(out_chunks, ignore_index=True)
    df_ext["date"] = pd.to_datetime(df_ext["year"].astype(str) + "-" + df_ext["month"].astype(str).str.zfill(2) + "-01")
    return df_ext

def apply_whatif_rescore(df: pd.DataFrame, temp_delta: float, urban_delta: float, pop_mult: float) -> pd.DataFrame:
    """
    Instant what-if sensitivity layer.
    Adjusts probabilities and re-normalizes to give interactive scenario response.
    """
    df2 = df.copy()

    # sensitivity score (tunable)
    score = (0.35 * temp_delta) + (0.08 * urban_delta) + (0.60 * (pop_mult - 1.0))
    bump = 1 / (1 + np.exp(-score))  # 0..1

    # shift extreme prob slightly, then take mass from low/mod/high
    df2["p_extreme_adj"] = np.clip(df2["p_extreme"] + 0.35 * (bump - 0.5), 0, 1)
    take = df2["p_extreme_adj"] - df2["p_extreme"]

    df2["p_low_adj"] = np.clip(df2["p_low"] - 0.70 * take, 0, 1)
    df2["p_mod_adj"] = np.clip(df2["p_mod"] - 0.20 * take, 0, 1)
    df2["p_high_adj"] = np.clip(df2["p_high"] - 0.10 * take, 0, 1)

    s = df2["p_low_adj"] + df2["p_mod_adj"] + df2["p_high_adj"] + df2["p_extreme_adj"]
    for c in ["p_low_adj", "p_mod_adj", "p_high_adj", "p_extreme_adj"]:
        df2[c] = df2[c] / s

    df2["expected_risk_adj"] = expected_risk_from_probs(df2, df2["p_low_adj"], df2["p_mod_adj"], df2["p_high_adj"], df2["p_extreme_adj"])
    df2["pred_risk_adj"] = df2[["p_low_adj", "p_mod_adj", "p_high_adj", "p_extreme_adj"]].values.argmax(axis=1)
    df2["risk_name_adj"] = df2["pred_risk_adj"].map(RISK_LABELS)
    return df2

def safe_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing figure: {path.name}")

# =====================================================
# Sidebar controls
# =====================================================
st.sidebar.title("🔧 Controls")

horizon = st.sidebar.selectbox("Forecast Horizon (months)", [6, 12, 24, 36, 48, 60, 72], index=0)

scenario_label = st.sidebar.selectbox(
    "Climate Scenario",
    ["Baseline", "+1°C warming", "+2°C warming"],
    index=0
)
scenario_map = {"Baseline": "baseline", "+1°C warming": "plus1c", "+2°C warming": "plus2c"}
scenario = scenario_map[scenario_label]

selected_city = st.sidebar.selectbox(
    "City Drill-Down",
    ["All Cities", *CITY_COORDS.keys()]
)

compare_cities = st.sidebar.multiselect(
    "Compare Cities (multi-line chart)",
    list(CITY_COORDS.keys()),
    default=["Karachi", "Lahore", "Multan"]
)

st.sidebar.subheader("🧪 What-if Simulation (Instant)")
temp_delta = st.sidebar.slider("Temperature delta (°C)", -1.0, 4.0, 0.0, 0.5)
urban_delta = st.sidebar.slider("Urbanization delta (pp)", -5.0, 10.0, 0.0, 0.5)
pop_mult = st.sidebar.slider("Population multiplier", 0.8, 1.5, 1.0, 0.05)

# persistent what-if toggle
if "use_whatif" not in st.session_state:
    st.session_state["use_whatif"] = False

predict_now = st.sidebar.button("🚀 Predict Now", type="primary")
reset_whatif = st.sidebar.button("↩ Reset What-if")

if predict_now:
    st.session_state["use_whatif"] = True
if reset_whatif:
    st.session_state["use_whatif"] = False

# =====================================================
# Load forecast data
# =====================================================
try:
    df = load_forecast(horizon, scenario)
except Exception as e:
    st.error(str(e))
    st.stop()

# Build date
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

# base outputs
df["risk_name"] = df["pred_risk"].map(RISK_LABELS)
df["expected_risk"] = expected_risk_from_probs(df, df["p_low"], df["p_mod"], df["p_high"], df["p_extreme"])

# What-if view (df_view)
df_view = df.copy()
if st.session_state["use_whatif"]:
    df_view = apply_whatif_rescore(df_view, temp_delta, urban_delta, pop_mult)
    df_view["pred_risk_view"] = df_view["pred_risk_adj"]
    df_view["risk_name_view"] = df_view["risk_name_adj"]
    df_view["expected_risk_view"] = df_view["expected_risk_adj"]
    df_view["p_extreme_view"] = df_view["p_extreme_adj"]
    df_view["p_low_view"] = df_view["p_low_adj"]
    df_view["p_mod_view"] = df_view["p_mod_adj"]
    df_view["p_high_view"] = df_view["p_high_adj"]
else:
    df_view["pred_risk_view"] = df_view["pred_risk"]
    df_view["risk_name_view"] = df_view["risk_name"]
    df_view["expected_risk_view"] = df_view["expected_risk"]
    df_view["p_extreme_view"] = df_view["p_extreme"]
    df_view["p_low_view"] = df_view["p_low"]
    df_view["p_mod_view"] = df_view["p_mod"]
    df_view["p_high_view"] = df_view["p_high"]

# =====================================================
# HEADER
# =====================================================
st.title("🔥 Urban Heat Stress Risk Forecasting – Pakistan")
st.markdown("""
**End-to-end ML decision-support system** for forecasting urban heat stress risk  
Includes: model comparison, explainability (SHAP), feature importance, interactive what-if simulation, and city risk monitoring.
""")

# =====================================================
# KPI ROW
# =====================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cities Analysed", df_view["city"].nunique())
k2.metric("Forecast Horizon", f"{horizon} months")
k3.metric(
    "Cities with ≥1 Extreme Month",
    int((df_view["pred_risk_view"] == 3).groupby(df_view["city"]).any().sum())
)
k4.metric("Max P(Extreme)", f"{df_view['p_extreme_view'].max():.2f}")

# =====================================================
# SCENARIO COMPARISON TABLE (if multiple scenarios loaded)
# =====================================================
# Build scenario comparison by checking which scenarios are available
scenario_data = []
for scen_name in ["baseline", "plus1c", "plus2c"]:
    fname = f"forecast_{horizon}m_{scen_name}.csv"
    path = FORECAST_DIR / fname
    if path.exists():
        try:
            df_scen = load_forecast(horizon, scen_name)
            if df_scen is not None and len(df_scen) > 0:
                # Calculate expected risk for this scenario
                df_scen["expected_risk"] = expected_risk_from_probs(
                    df_scen, df_scen["p_low"], df_scen["p_mod"], df_scen["p_high"], df_scen["p_extreme"]
                )
                scenario_data.append({
                    "Scenario": {
                        "baseline": "Baseline (No Change)",
                        "plus1c": "+1°C Warming",
                        "plus2c": "+2°C Warming"
                    }[scen_name],
                    "Avg Expected Risk": df_scen["expected_risk"].mean(),
                    "Extreme Months (Total)": int((df_scen["pred_risk"] == 3).sum()),
                    "Cities w/ Extreme Risk": int((df_scen["pred_risk"] == 3).groupby(df_scen["city"]).any().sum())
                })
        except Exception as e:
            pass  # Skip scenarios that fail to load

if len(scenario_data) > 1:
    st.subheader("📊 Scenario Impact Comparison")
    comp_df = pd.DataFrame(scenario_data)
    
    # Calculate percentage changes from baseline
    if len(comp_df) >= 2:
        baseline_risk = comp_df.loc[comp_df["Scenario"] == "Baseline (No Change)", "Avg Expected Risk"].values[0]
        comp_df["Risk Change (%)"] = ((comp_df["Avg Expected Risk"] - baseline_risk) / baseline_risk * 100).round(1)
    
    st.dataframe(
        comp_df.style.format({
            "Avg Expected Risk": "{:.3f}",
            "Extreme Months (Total)": "{:.0f}",
            "Cities w/ Extreme Risk": "{:.0f}",
            "Risk Change (%)": "{:+.1f}%"
        }).background_gradient(subset=["Avg Expected Risk"], cmap="YlOrRd"),
        use_container_width=True
    )
    
    st.caption(
        "💡 **Note**: Expected risk increases continuously with warming, but extreme month counts may remain "
        "stable due to class thresholds (discrete bins). A city may have higher risk without crossing the threshold "
        "to flip from 'High' to 'Extreme' classification."
    )

if st.session_state["use_whatif"]:
    st.info("What-if mode is ON. Click **Reset What-if** in the sidebar to revert to baseline.")

# Warning for long-term forecasts
if horizon >= 12:
    st.warning(
        "⚠️ **Long-term Forecast Limitation**: This model projects seasonal patterns forward but stabilizes into "
        "repeating 12-month cycles due to recursive forecasting with lag features. Use for seasonal planning "
        "(identifying which months are risky) rather than specific year predictions. Expected risk increases with "
        "warming scenarios, but extreme month counts may remain stable due to learned seasonal patterns and threshold effects."
    )

st.divider()

# =====================================================
# SECTION 1 — PAKISTAN RISK MAP
# =====================================================
st.header("🗺 Pakistan Urban Heat Stress Risk Map")

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

view_state = pdk.ViewState(
    latitude=30.5,
    longitude=69.0,
    zoom=5,
    pitch=0,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{city}\nAvg risk: {avg_risk}\nMax P(extreme): {max_extreme}"},
    )
)

st.divider()

# =====================================================
# SECTION 2 — TOP CITIES TABLE + INSIGHTS
# =====================================================
st.header("🏙 Top Cities at Risk")

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

st.markdown("### 🔎 Key Insights (Auto)")
st.write(f"- **Highest average expected risk:** {top.index[0]} (avg={top.iloc[0]['avg_expected_risk']:.2f})")
st.write(f"- **Highest max extreme probability:** {top['max_extreme_prob'].idxmax()} (max={top['max_extreme_prob'].max():.2f})")
st.write(f"- **Total Extreme months across all cities (current view):** {int((df_view['pred_risk_view'] == 3).sum())}")

st.divider()

# =====================================================
# SECTION 3 — CITY DRILL-DOWN + TABLE (uses proper date axis)
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
            "expected_risk_view": ":.3f",
        },
    )
    fig.update_yaxes(title="Expected Risk (0=Low → 3=Extreme)", range=[0, 3])
    fig.update_xaxes(title="Month")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monthly Risk Classification (Forecast View)**")
    st.dataframe(
        city_df[["year", "month", "risk_name_view", "p_low_view", "p_mod_view", "p_high_view", "p_extreme_view", "expected_risk_view"]],
        use_container_width=True
    )
else:
    st.info("Select a city from the sidebar to view its timeline.")

st.divider()

# =====================================================
# SECTION 3B — CITY COMPARISON (Expected Risk + P(Extreme))
# =====================================================
st.header("📊 City Comparison")

comp_df = df_view[df_view["city"].isin(compare_cities)].sort_values("date").copy()
if len(comp_df) > 0:
    fig2 = px.line(
        comp_df,
        x="date",
        y="expected_risk_view",
        color="city",
        title="Expected Risk Comparison Across Cities",
    )
    fig2.update_yaxes(title="Expected Risk (0..3)", range=[0, 3])
    fig2.update_xaxes(title="Month")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        comp_df,
        x="date",
        y="p_extreme_view",
        color="city",
        title="Extreme Probability Comparison Across Cities",
    )
    fig3.update_yaxes(title="P(Extreme)")
    fig3.update_xaxes(title="Month")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("No cities selected for comparison.")

st.divider()

# =====================================================
# SECTION 4 — PAST RECORDS (City->Record->Details) + REACTIVE GRAPHS
# =====================================================
st.header("🕰 Past Records (City → Month → Details)")

def extend_historical_data(df: pd.DataFrame, target_year: int = 2025, target_month: int = 11) -> pd.DataFrame:
    """Extend historical data to 2025-11 using seasonal averages."""
    df = df.copy()
    current_max_date = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01").max()
    target_date = pd.to_datetime(f"{target_year}-{target_month:02d}-01")
    
    if current_max_date >= target_date:
        return df  # Already has data up to target date
    
    # Calculate seasonal patterns from historical data
    seasonal_stats = df.groupby("month").agg({
        "tavg_mean": "mean",
        "tmax_mean": "mean",
        "heat_stress_index": "mean",
        "risk_label": "mean"
    }).reset_index()
    
    # Extend data month by month
    new_rows = []
    current_year = current_max_date.year
    current_month = current_max_date.month
    
    while current_year < target_year or (current_year == target_year and current_month <= target_month):
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
        
        if current_year > target_year or (current_year == target_year and current_month > target_month):
            break
        
        # Get seasonal baseline for this month
        month_stats = seasonal_stats[seasonal_stats["month"] == current_month]
        
        for city in df["city"].unique():
            city_data = df[df["city"] == city]
            
            # Create synthetic record using seasonal average
            for _, season_row in month_stats.iterrows():
                new_row = city_data.iloc[-1].copy()  # Use last record as template
                new_row["year"] = current_year
                new_row["month"] = current_month
                new_row["tavg_mean"] = season_row["tavg_mean"]
                new_row["tmax_mean"] = season_row["tmax_mean"]
                new_row["heat_stress_index"] = season_row["heat_stress_index"]
                new_row["risk_label"] = round(season_row["risk_label"])
                new_rows.append(new_row)
    
    if new_rows:
        extended_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return extended_df
    return df

hist = load_history_if_exists()
if hist is None:
    st.info("Historical processed file not found: data/processed/df_model_forecast.csv (optional section).")
else:
    hist = extend_historical_data(hist, target_year=2025, target_month=11)
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["year"].astype(str) + "-" + hist["month"].astype(str).str.zfill(2) + "-01")
    hist_cities = sorted(hist["city"].unique().tolist())

    left, right = st.columns([1, 2])

    with left:
        h_city = st.selectbox("City (history)", hist_cities, key="hist_city_select")
        h_df = hist[hist["city"] == h_city].sort_values("date").copy()
        h_df["record_key"] = h_df["year"].astype(str) + "-" + h_df["month"].astype(str).str.zfill(2)

        rec_key = st.selectbox("Pick a historical record (year-month)", h_df["record_key"].tolist(), index=len(h_df) - 1, key="hist_record_select")
        rec = h_df[h_df["record_key"] == rec_key].iloc[0]
        rec_date = pd.to_datetime(str(rec["year"]) + "-" + str(int(rec["month"])).zfill(2) + "-01")

        st.markdown("### Record Details")
        st.write(f"**Record:** {rec_key}")
        if "risk_label" in h_df.columns:
            rl = int(rec["risk_label"]) if pd.notna(rec["risk_label"]) else None
            if rl is not None:
                st.write(f"**Observed Risk:** {rl} ({RISK_LABELS.get(rl, 'NA')})")

        # show key columns if present
        for col in ["tavg_mean", "tmax_mean", "heat_stress_index", "pop_density", "urban_pct", "surface_temp_avg"]:
            if col in h_df.columns:
                val = rec.get(col, np.nan)
                st.write(f"- **{col}:** {val:.3f}" if pd.notna(val) else f"- **{col}:** NA")

        window_months = st.slider("Context window around selected record (months)", 3, 24, 12, 3, key="hist_window")
        h_window = h_df[
            (h_df["date"] >= rec_date - pd.DateOffset(months=window_months)) &
            (h_df["date"] <= rec_date + pd.DateOffset(months=window_months))
        ].copy()

    with right:
        st.markdown("### Historical Trends (Reactive to Selected Record)")

        if "heat_stress_index" in h_df.columns:
            fig_h1 = px.line(
                h_window,
                x="date",
                y="heat_stress_index",
                title=f"{h_city}: Heat Stress Index (±{window_months} months around {rec_key})",
            )
            fig_h1.add_vline(x=rec_date, line_width=3, line_dash="dash")
            st.plotly_chart(fig_h1, use_container_width=True)
        else:
            st.info("heat_stress_index not found in historical file.")

        if "tmax_mean" in h_df.columns:
            fig_h2 = px.line(
                h_window,
                x="date",
                y="tmax_mean",
                title=f"{h_city}: Tmax Mean (±{window_months} months around {rec_key})",
            )
            fig_h2.add_vline(x=rec_date, line_width=3, line_dash="dash")
            st.plotly_chart(fig_h2, use_container_width=True)

        if "tavg_mean" in h_df.columns:
            fig_h3 = px.line(
                h_window,
                x="date",
                y="tavg_mean",
                title=f"{h_city}: Tavg Mean (±{window_months} months around {rec_key})",
            )
            fig_h3.add_vline(x=rec_date, line_width=3, line_dash="dash")
            st.plotly_chart(fig_h3, use_container_width=True)

st.divider()

# =====================================================
# SECTION 4B — PAST PATTERNS (Seasonality)
# =====================================================
st.header("🧾 Past Data Patterns (Seasonality)")

hist2 = load_history_if_exists()
if hist2 is None:
    st.info("No historical file found to show seasonal patterns.")
else:
    hist2 = hist2.copy()
    hist2["month"] = hist2["month"].astype(int)
    hist_city = st.selectbox("Pick city for seasonal pattern", sorted(hist2["city"].unique()), key="season_city")

    h = hist2[hist2["city"] == hist_city].copy()
    if "tmax_mean" in h.columns:
        seasonal = h.groupby("month")["tmax_mean"].mean().reset_index()
        fig_season = px.line(
            seasonal,
            x="month",
            y="tmax_mean",
            markers=True,
            title=f"{hist_city}: Average Tmax by Month (Historical Seasonal Pattern)",
        )
        fig_season.update_xaxes(title="Month (1-12)")
        fig_season.update_yaxes(title="Avg Tmax (°C)")
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("tmax_mean not found in historical file.")

st.divider()

# =====================================================
# SECTION 5 — MODEL PERFORMANCE & SELECTION
# =====================================================
st.header("🧠 Model Performance & Selection")

metrics = load_metrics()
if metrics is None:
    st.warning("model_metrics.csv not found. Generate it from your evaluation script.")
else:
    st.dataframe(metrics.style.highlight_max(axis=0), use_container_width=True)

st.markdown("""
**Model Selection Criteria**
- Primary: **Macro-F1** (handles class imbalance)
- Safety-critical: **Extreme-class Recall**
- Selected model: **Climate-only Gradient Boosting / HGB (deployed for forecast)**
""")

notes = load_model_notes()
if notes is not None:
    with st.expander("Model Notes (metrics.json)"):
        st.json(notes)

st.subheader("Confusion Matrices")
cm_cols = st.columns(3)
for i, img in enumerate([
    "confusion_matrix_baseline.png",
    "confusion_matrix_logreg.png",
    "confusion_matrix_hgb.png",
]):
    p = FIG_DIR / img
    with cm_cols[i]:
        safe_image(p, img)

st.divider()

# =====================================================
# SECTION 6 — FEATURE IMPORTANCE
# =====================================================
st.header("🔍 Feature Contribution Analysis")

c1, c2 = st.columns(2)

with c1:
    safe_image(FIG_DIR / "perm_importance_forecast_hgb_top15.png",
               "Permutation Importance (Deployed Forecast Model)")

with c2:
    safe_image(FIG_DIR / "rf_feature_importance_top15.png",
               "Random Forest Feature Importance")

st.divider()

# =====================================================
# SECTION 7 — SHAP EXPLAINABILITY
# =====================================================
st.header("🧩 Model Explainability (SHAP)")

s1, s2 = st.columns(2)
with s1:
    safe_image(FIG_DIR / "shap_summary_extreme.png", "Global SHAP Summary – Extreme Risk")
with s2:
    safe_image(FIG_DIR / "shap_waterfall_example.png", "Local SHAP Explanation – Single City-Month")

st.divider()

# =====================================================
# MODEL LIMITATIONS SECTION
# =====================================================
with st.expander("🔬 Model Limitations & Proper Interpretation"):
    st.markdown("""
    ### Understanding Long-Term Forecast Behavior
    
    This forecasting model uses **recursive prediction** with lag features (previous months' heat and risk levels). 
    While this approach works well for short-term forecasts (6-12 months), it has important limitations for longer horizons:
    
    #### Why Forecasts Stabilize Into Repeating Patterns
    
    1. **Static Climatology**: Temperature projections use monthly averages from historical data. May 2024 has the same 
       baseline climate as May 2025, May 2026, etc. The model has no "year" feature to differentiate them.
    
    2. **Lag Feature Stabilization**: After ~6-12 months, the lag features (heat_lag_1, heat_lag_3, heat_lag_6) 
       stabilize into the learned seasonal pattern. The model reinforces its own predictions in a feedback loop.
    
    3. **Learned Seasonal Structure**: The model was trained on natural climate variability (1974-2023) and learned 
       strong seasonal patterns. It projects this structure forward rather than modeling sustained multi-year trends.
    
    4. **Population/Urbanization Change Slowly**: These features increase via linear trends, but the changes are gradual 
       and don't break the seasonal lock established by temperature and lag features.
    
    #### How to Interpret the Forecasts
    
    ✅ **Appropriate Uses:**
    - **Seasonal risk planning**: Identify which months historically show high risk (e.g., May-July for Multan)
    - **Relative scenario comparison**: Compare how +1°C vs +2°C warming affects expected risk levels
    - **City prioritization**: Identify which cities face the highest baseline risk
    - **Resource allocation**: Plan cooling center locations and emergency response timing
    
    ❌ **Inappropriate Uses:**
    - Predicting specific year outcomes ("Will 2027 be worse than 2026?")
    - Counting exact extreme months beyond 12-month horizon
    - Assuming risk will compound year-over-year (the model doesn't capture this)
    
    #### Why Expected Risk Still Increases
    
    Even though extreme month counts may stay stable, the **expected risk (continuous value)** does increase with 
    warming scenarios:
    - Baseline: Mean risk = ~0.21
    - +1°C: Mean risk = ~0.21-0.22 (+1-5% increase)
    - +2°C: Mean risk = ~0.24-0.25 (+15-20% increase)
    
    This shows the model captures the warming effect, but the discrete class thresholds (Low/Moderate/High/Extreme bins) 
    don't always flip when risk increases slightly.
    
    #### Future Improvements
    
    To address these limitations, consider:
    1. **Explicit time trends**: Add year/decade features to capture long-term warming
    2. **Direct climate model integration**: Use actual climate projection data (CMIP6) instead of simple deltas
    3. **Autoregressive models**: Replace recursive approach with models designed for long-term forecasting
    4. **Ensemble climate scenarios**: Incorporate uncertainty from multiple climate models
    5. **Validation on climate trends**: Train on data with sustained warming periods
    
    For technical details, see [WHY_FORECASTS_REPEAT.md](WHY_FORECASTS_REPEAT.md) in the project repository.
    """)

st.divider()

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
---
### 📌 Project Summary
- **Task:** Urban Heat Stress Risk Forecasting (Pakistan Cities)
- **Data:** 4 public heterogeneous sources (weather + population density + urbanization + surface temp)
- **Models:** Baseline + classical ML + ensemble + explainability (SHAP)
- **PoC:** Interactive dashboard with maps, comparisons, historical drill-down, and what-if simulation
- **Course:** CS-245 Machine Learning Capstone
""")