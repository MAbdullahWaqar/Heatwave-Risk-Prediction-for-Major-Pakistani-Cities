import os
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path
import plotly.express as px

SEQUENCE_CHECKPOINT_NAME = os.environ.get("SEQUENCE_CHECKPOINT_NAME", "gru_attn_best.pkl")

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
            f"From the `heat-risk-pk` folder run:\n"
            f"  python -m src.forecast\n"
            f"(uses **GRU** weights `models/{SEQUENCE_CHECKPOINT_NAME}`; see `src/config.py` / `src/forecast_lstm.py`.)"
        )
    return pd.read_csv(path)

@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    path = FIG_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_model_selection_scores() -> pd.DataFrame:
    """
    Preferred source: notebook export if available.
    Fallback: known architecture comparison scores from training run.
    """
    p1 = FIG_DIR / "gru_training_summary.csv"
    p2 = ROOT / "models" / "deep_learning" / "dl_model_comparison.csv"
    for p in (p1, p2):
        if p.exists():
            df = pd.read_csv(p)
            if {"model", "best_val_macro_f1"}.issubset(df.columns):
                return df[["model", "best_val_macro_f1"]].sort_values(
                    "best_val_macro_f1", ascending=False
                ).reset_index(drop=True)

    return pd.DataFrame(
        {
            "model": ["GRU_Attn", "LSTM_Attn", "TCN", "Transformer"],
            "best_val_macro_f1": [0.906831, 0.879589, 0.871994, 0.286878],
        }
    )


@st.cache_data
def load_history_if_exists() -> pd.DataFrame | None:
    path = DATA_PROCESSED / "df_model_forecast.csv"
    return pd.read_csv(path) if path.exists() else None

def expected_risk_from_probs(df: pd.DataFrame, p_low, p_mod, p_high, p_extreme) -> pd.Series:
    return 0*p_low + 1*p_mod + 2*p_high + 3*p_extreme


def expected_risk_to_rgb(avg: float) -> tuple[int, int, int]:
    """Continuous 0–3 expected risk → RGB (cool/green → yellow → hot/red). Used on map so what-if shifts show even when argmax class stays Extreme."""
    t = float(np.clip(float(avg) / 3.0, 0.0, 1.0))
    r = int(35 + 215 * (t**1.05))
    g = int(195 * ((1.0 - t) ** 1.1))
    b = int(28 + 35 * t)
    return (min(r, 255), min(max(g, 0), 255), min(b, 255))

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

def _ensure_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce GRU forecast class probabilities to finite floats and renormalize per row."""
    out = df.copy()
    for c in ("p_low", "p_mod", "p_high", "p_extreme"):
        if c not in out.columns:
            raise ValueError(f"Forecast CSV missing required column `{c}` for what-if (GRU softmax outputs).")
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = out["p_low"] + out["p_mod"] + out["p_high"] + out["p_extreme"]
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-12)
    for c in ("p_low", "p_mod", "p_high", "p_extreme"):
        out[c] = out[c] / s
    return out


def _city_vulnerability_factors(df_hist: pd.DataFrame | None, cities: pd.Series) -> pd.Series:
    """
    Build a per-city multiplier so what-if slider changes do not affect all cities equally.
    Uses available historical context; falls back to 1.0 if history is missing.
    """
    if df_hist is None or df_hist.empty:
        return pd.Series(1.0, index=cities.index)

    h = df_hist.copy()
    if "city" not in h.columns:
        return pd.Series(1.0, index=cities.index)
    h["city"] = h["city"].astype(str).str.strip()

    use_cols = [c for c in ["urban_pct", "pop_density", "heat_stress_index"] if c in h.columns]
    if not use_cols:
        return pd.Series(1.0, index=cities.index)

    city_ctx = h.groupby("city")[use_cols].mean(numeric_only=True)
    score = pd.Series(0.0, index=city_ctx.index)
    if "urban_pct" in city_ctx.columns:
        u = city_ctx["urban_pct"]
        score += 0.45 * ((u - u.mean()) / (u.std(ddof=0) + 1e-9))
    if "pop_density" in city_ctx.columns:
        p = city_ctx["pop_density"]
        score += 0.35 * ((p - p.mean()) / (p.std(ddof=0) + 1e-9))
    if "heat_stress_index" in city_ctx.columns:
        hs = city_ctx["heat_stress_index"]
        score += 0.20 * ((hs - hs.mean()) / (hs.std(ddof=0) + 1e-9))

    # 0.75..1.45 roughly; clips prevent unstable overreaction.
    factor = (1.0 + 0.30 * np.tanh(score)).clip(0.75, 1.45)
    return cities.map(factor).fillna(1.0)


def apply_whatif_rescore(
    df: pd.DataFrame,
    temp_delta: float,
    urban_delta: float,
    pop_mult: float,
    humidity_delta: float,
    precip_delta: float,
    vegetation_delta: float,
    wind_delta: float,
    df_hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Instant what-if sensitivity layer on top of **GRU** forecast class probabilities (`p_*`).
    Heuristic only — does not re-run the torch model. Stronger than before so sliders move
    `expected_risk` and often `pred_risk`, not just tiny softmax nudges.
    """
    df2 = _ensure_prob_columns(df)

    # Base stress from user controls; signs reflect general heat-risk tendency:
    # temp/humidity/urban/pop increase risk, while precipitation/vegetation/wind reduce risk.
    base_stress = (
        0.52 * float(temp_delta)
        + 0.07 * float(urban_delta)
        + 0.92 * (float(pop_mult) - 1.0)
        + 0.030 * float(humidity_delta)
        - 0.020 * float(precip_delta)
        - 1.20 * float(vegetation_delta)
        - 0.050 * float(wind_delta)
    )

    city_factor = _city_vulnerability_factors(df_hist, df2["city"])
    if "heat_stress_index_proj" in df2.columns:
        hs = pd.to_numeric(df2["heat_stress_index_proj"], errors="coerce").fillna(0.0)
        hs_z = (hs - hs.mean()) / (hs.std(ddof=0) + 1e-9)
        local_amp = (1.0 + 0.12 * np.tanh(hs_z)).clip(0.85, 1.20)
    else:
        local_amp = 1.0

    stress = np.tanh(0.58 * base_stress * city_factor * local_amp)
    df2["p_extreme_adj"] = np.clip(df2["p_extreme"] + 0.52 * stress, 0.0, 1.0)
    take = df2["p_extreme_adj"] - df2["p_extreme"]

    df2["p_low_adj"] = np.clip(df2["p_low"] - 0.82 * take, 0.0, 1.0)
    df2["p_mod_adj"] = np.clip(df2["p_mod"] - 0.12 * take, 0.0, 1.0)
    df2["p_high_adj"] = np.clip(df2["p_high"] - 0.06 * take, 0.0, 1.0)

    s = df2["p_low_adj"] + df2["p_mod_adj"] + df2["p_high_adj"] + df2["p_extreme_adj"]
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-12)
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
st.sidebar.title("Controls")

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

st.sidebar.subheader("What-if on GRU outputs (instant)")
st.sidebar.caption(
    "Rescores **saved** GRU class probabilities with climate/urban/eco controls. "
    "After **Predict Now**, move temperature, humidity, precipitation, vegetation and other sliders; "
    "city risk updates immediately in map/table/charts. **Reset** turns what-if off."
)

if "use_whatif" not in st.session_state:
    st.session_state["use_whatif"] = False

# Reset must run *before* sliders: widget keys cannot be assigned after the widget is created.
reset_whatif = st.sidebar.button("↩ Reset What-if")
if reset_whatif:
    st.session_state["use_whatif"] = False
    for _wk in (
        "whatif_temp_delta",
        "whatif_urban_delta",
        "whatif_pop_mult",
        "whatif_humidity_delta",
        "whatif_precip_delta",
        "whatif_vegetation_delta",
        "whatif_wind_delta",
    ):
        st.session_state.pop(_wk, None)
    st.rerun()

temp_delta = st.sidebar.slider(
    "Temperature delta (°C)",
    -1.0,
    4.0,
    0.0,
    0.5,
    key="whatif_temp_delta",
    help="Heuristic stress tilt on top of GRU probs.",
)
urban_delta = st.sidebar.slider(
    "Urbanization delta (pp)", -5.0, 10.0, 0.0, 0.5, key="whatif_urban_delta"
)
pop_mult = st.sidebar.slider(
    "Population multiplier", 0.8, 1.5, 1.0, 0.05, key="whatif_pop_mult"
)
humidity_delta = st.sidebar.slider(
    "Humidity delta (%)", -20.0, 30.0, 0.0, 1.0, key="whatif_humidity_delta"
)
precip_delta = st.sidebar.slider(
    "Precipitation delta (%)", -60.0, 80.0, 0.0, 2.0, key="whatif_precip_delta"
)
vegetation_delta = st.sidebar.slider(
    "Vegetation (NDVI) delta", -0.30, 0.30, 0.0, 0.01, key="whatif_vegetation_delta"
)
wind_delta = st.sidebar.slider(
    "Wind speed delta (m/s)", -3.0, 3.0, 0.0, 0.1, key="whatif_wind_delta"
)

predict_now = st.sidebar.button("Predict Now", type="primary")
if predict_now:
    st.session_state["use_whatif"] = True

_whatif_active = bool(st.session_state.get("use_whatif", False))

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

# GRU softmax columns: coerce so what-if and KPIs never see NaN/object dtypes from CSV
for _c in ("p_low", "p_mod", "p_high", "p_extreme"):
    if _c in df.columns:
        df[_c] = pd.to_numeric(df[_c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
_sprob = df["p_low"] + df["p_mod"] + df["p_high"] + df["p_extreme"]
_sprob = np.maximum(np.asarray(_sprob, dtype=np.float64), 1e-12)
for _c in ("p_low", "p_mod", "p_high", "p_extreme"):
    df[_c] = df[_c] / _sprob

# base outputs
df["risk_name"] = df["pred_risk"].map(RISK_LABELS)
df["expected_risk"] = expected_risk_from_probs(df, df["p_low"], df["p_mod"], df["p_high"], df["p_extreme"])

# What-if view (df_view): after **Predict Now**, heuristic rescoring of GRU probs using slider deltas
df_view = df.copy()
if _whatif_active:
    df_view = apply_whatif_rescore(
        df_view,
        temp_delta,
        urban_delta,
        pop_mult,
        humidity_delta,
        precip_delta,
        vegetation_delta,
        wind_delta,
        df_hist=load_history_if_exists(),
    )
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
st.title(" Urban Heat Stress Risk Forecasting – Pakistan")
st.markdown("""
**Deep learning decision-support system** — bidirectional **GRU + attention** over multi-month sequences (default weights **`gru_attn_best.pkl`**) for urban heat stress risk.  
Forecasts come from the **GRU**; the dashboard adds scenario comparison, what-if rescoring, and city-level monitoring.
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

st.caption(
    "💡 **Note**: Expected risk increases continuously with warming, but extreme month counts may remain "
    "stable due to class thresholds (discrete bins). A city may have higher risk without crossing the threshold "
    "to flip from 'High' to 'Extreme' classification."
)

if _whatif_active:
    st.info(
        "**What-if mode** — **heuristic** rescoring of the GRU’s saved class probabilities using your slider values. "
        "Click **Reset What-if** in the sidebar to turn off what-if and return sliders to neutral (0 °C, 0 pp, 1.0× pop)."
    )

# Warning for long-term forecasts
if horizon >= 12:
    st.warning(
        "⚠️ **Long-term forecast limitation**: The **GRU** is rolled forward month-by-month with projected climate and "
        "recursive heat-index features, so long horizons tend toward repeating seasonal structure. Use for **which months** "
        "are riskiest and for **scenario deltas** (+1°C / +2°C), not for exact year-to-year ranking."
    )

st.divider()

# =====================================================
# SECTION 1 — PAKISTAN RISK MAP
# =====================================================
st.header("Pakistan Urban Heat Stress Risk Map")

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
# Colour by **continuous** mean expected risk — not worst discrete class — so what-if and prob shifts show visually.
_rgb = map_df["avg_risk"].astype(float).map(expected_risk_to_rgb)
map_df["r"] = _rgb.map(lambda x: x[0])
map_df["g"] = _rgb.map(lambda x: x[1])
map_df["b"] = _rgb.map(lambda x: x[2])
map_df["a"] = 235
# Geographic radius (meters) for data-driven *relative* size; **radius_max_pixels** caps on-screen blob size
# (PyDeck / deck.gl often defaults to meters — huge pixel radii were drawn as giant red plates).
map_df["circle_radius_m"] = (
    1200.0 + map_df["max_extreme"].astype(float) * 8000.0 + map_df["avg_risk"].astype(float) * 900.0
).clip(800.0, 12000.0)

layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position=["lon", "lat"],
    get_radius="circle_radius_m",
    get_fill_color="[r, g, b, a]",
    pickable=True,
    radius_units="meters",
    radius_min_pixels=4,
    radius_max_pixels=10,
    stroked=True,
    get_line_color=[25, 25, 25, 180],
    line_width_min_pixels=1,
    opacity=0.95,
)

view_state = pdk.ViewState(
    latitude=30.5,
    longitude=69.0,
    zoom=5,
    pitch=0,
)

_deck_key = (
    f"deck_{horizon}_{scenario}_{_whatif_active}_"
    f"{temp_delta:.4f}_{urban_delta:.4f}_{pop_mult:.4f}_"
    f"{humidity_delta:.4f}_{precip_delta:.4f}_{vegetation_delta:.4f}_{wind_delta:.4f}_"
    f"{map_df['circle_radius_m'].sum():.1f}"
)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "text": (
                "{city}\n"
                "Avg expected risk (0–3): {avg_risk}\n"
                "Max P(extreme): {max_extreme}\n"
                "Worst predicted bin (0–3): {risk}"
            )
        },
    ),
    key=_deck_key,
)
st.caption(
    "**Colour** = **average expected risk** (probability-weighted 0–3), so it updates under what-if even when the worst **class** stays Extreme. "
    "Dot size is capped (~4–10 px). **Scenario** dropdown loads different GRU CSVs (+1 °C etc.); what-if is a fast heuristic on the current file."
)

st.divider()

# =====================================================
# SECTION 2 — TOP CITIES TABLE + INSIGHTS (GRU)
# =====================================================
st.header("Top cities at risk (GRU)")
st.caption(
    "Ranked from **GRU** forecast rows for the selected **horizon** and **climate scenario** "
    + ("(with **what-if** rescoring applied)." if _whatif_active else "(loaded CSV).")
)

top = (
    df_view.groupby("city")
    .agg(
        avg_expected_risk=("expected_risk_view", "mean"),
        max_extreme_prob=("p_extreme_view", "max"),
        extreme_months=("pred_risk_view", lambda s: int((s == 3).sum())),
    )
    .sort_values("avg_expected_risk", ascending=False)
)

FIG_DIR.mkdir(parents=True, exist_ok=True)
_top_cities_path = FIG_DIR / f"top_cities_gru_{horizon}m_{scenario}.csv"
top.to_csv(_top_cities_path)
st.caption(f"Written: **`{_top_cities_path.relative_to(ROOT)}`** (refreshes when you change horizon, scenario, or what-if).")

st.dataframe(top.style.background_gradient(cmap="Reds"), use_container_width=True)
st.download_button(
    label="Download top cities (GRU CSV)",
    data=top.to_csv().encode("utf-8"),
    file_name=f"top_cities_gru_{horizon}m_{scenario}.csv",
    mime="text/csv",
    key="dl_top_cities_gru",
)

st.markdown("### Key Insights (Auto)")
st.write(f"- **Highest average expected risk:** {top.index[0]} (avg={top.iloc[0]['avg_expected_risk']:.2f})")
st.write(f"- **Highest max extreme probability:** {top['max_extreme_prob'].idxmax()} (max={top['max_extreme_prob'].max():.2f})")
st.write(f"- **Total Extreme months across all cities (current view):** {int((df_view['pred_risk_view'] == 3).sum())}")

st.divider()

# =====================================================
# SECTION 3 — CITY DRILL-DOWN + TABLE (uses proper date axis)
# =====================================================
st.header("City Risk Timeline")

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
st.header("City Comparison")

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
st.header("Past Records (City → Month → Details)")

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
st.header("Past Data Patterns (Seasonality)")

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
st.header("Model Performance & Selection")

metrics = load_metrics()
if metrics is None:
    st.warning(
        "`outputs/figures/model_metrics.csv` not found. From `heat-risk-pk` run: `python -m src.evaluate` "
        f"(evaluates **GRU** `models/{SEQUENCE_CHECKPOINT_NAME}` on the held-out test period)."
    )
else:
    st.subheader("Test metrics (`model_metrics.csv`)")
    metrics_view = metrics.copy()
    if "model" in metrics_view.columns:
        metrics_view = metrics_view[metrics_view["model"] != "baseline_majority"].reset_index(drop=True)
    st.dataframe(metrics_view, use_container_width=True)

st.subheader("Model selection candidates (validation macro-F1)")
sel_df = load_model_selection_scores()
st.dataframe(
    sel_df.style.format({"best_val_macro_f1": "{:.6f}"}).background_gradient(
        subset=["best_val_macro_f1"], cmap="YlOrRd"
    ),
    use_container_width=True,
)

if "forecast_model" in df.columns:
    ck = str(df["forecast_checkpoint"].iloc[0]) if "forecast_checkpoint" in df.columns else ""
    st.caption(f"Loaded forecast rows: **{df['forecast_model'].iloc[0]}** (`{ck}`).")

st.markdown("""
**GRU (sequence model)**  
- **Architecture:** bidirectional **GRU** + attention pooling + city embedding → softmax over 4 risk classes (`RNNAttentionClassifier` in `src/lstm_risk_model.py`).  
- **Inputs:** `seq_len` consecutive months of scaled numeric features per city (same schema as `notebooks/deep_learning_model_selection.ipynb`).  
- **Training:** validation **macro-F1** with early stopping in the notebook; **test** metrics in the table above from `python -m src.evaluate`.  
- **Artifacts:** **`models/gru_attn_best.pkl`** (via **`SEQUENCE_CHECKPOINT_NAME`**). Forecasts: `python -m src.forecast` (`src/forecast_lstm.py`).
""")

st.subheader("Confusion matrix (GRU, test set)")
_cm_path = FIG_DIR / "confusion_matrix_sequence.png"
safe_image(_cm_path, "GRU + Attention — normalized confusion matrix (held-out years)")

st.divider()

# =====================================================
# SECTION 6 — FEATURE IMPORTANCE (GRU)
# =====================================================
st.header("Feature contribution (GRU)")

_sal_path = FIG_DIR / "sequence_feature_saliency_top15.png"
if not _sal_path.exists():
    _sal_path = FIG_DIR / "lstm_feature_saliency_top15.png"
safe_image(
    _sal_path,
    "Input × gradient saliency on the **GRU** (top features, test subsample — run `python -m src.evaluate` to refresh).",
)

st.caption(
    "Optional **Kernel SHAP** on the same torch model: run the SHAP cells in `notebooks/deep_learning_model_selection.ipynb`."
)

st.divider()

# =====================================================
# SECTION 7 — SHAP EXPLAINABILITY (GRU)
# =====================================================
st.header("Model Explainability (SHAP)")
st.caption(
    "Kernel SHAP visualizations for the GRU model. Generate/update with "
    "`cd heat-risk-pk && python -m src.generate_shap`."
)

_shap_summary = FIG_DIR / "shap_summary_extreme.png"
_shap_waterfall = FIG_DIR / "shap_waterfall_example.png"
_shap_table = FIG_DIR / "shap_feature_importance_best_model.csv"

scol1, scol2 = st.columns(2)
with scol1:
    safe_image(_shap_summary, "SHAP Summary (Extreme risk class)")
with scol2:
    safe_image(_shap_waterfall, "SHAP Waterfall (example city-month)")

if _shap_table.exists():
    st.markdown("#### SHAP Feature Ranking (Top Rows)")
    st.dataframe(pd.read_csv(_shap_table).head(20), use_container_width=True)
else:
    st.info(
        "Missing SHAP table `shap_feature_importance_best_model.csv`. "
        "Run the notebook SHAP cell to create it."
    )

st.divider()

# =====================================================
# MODEL LIMITATIONS SECTION
# =====================================================
with st.expander("Model Limitations & Proper Interpretation"):
    st.markdown("""
    ### Understanding Long-Term Forecast Behavior
    
    The **GRU** is applied in **recursive** mode for forecasts: each new month updates the sliding window of scaled features 
    and heat-stress lags. That works well for near-term horizons; long horizons inherit limitations:
    
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
    
   
    """)

st.divider()

