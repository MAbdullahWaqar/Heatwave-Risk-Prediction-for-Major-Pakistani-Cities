import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from .config import MODELS_DIR, FORECAST_DIR, DATA_PROCESSED


def next_year_month(y, m):
    m2 = m + 1
    y2 = y
    if m2 == 13:
        m2 = 1
        y2 += 1
    return y2, m2


def z(x, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return (x - mean) / std


def compute_heat_index(row, stats):
    tavg_z = z(row["tavg_mean"], *stats["tavg_mean"])
    tmax_z = z(row["tmax_mean"], *stats["tmax_mean"])
    surf_z = z(row["surface_temp_avg"], *stats["surface_temp_avg"])
    pop_z  = z(row["pop_density"], *stats["pop_density"])
    urb_z  = z(row["urban_pct"], *stats["urban_pct"])
    wind_z = z(row["wspd_mean_filled"], *stats["wspd_mean_filled"])
    return (0.40*tavg_z + 0.30*tmax_z + 0.15*surf_z + 0.10*pop_z + 0.10*urb_z - 0.05*wind_z)


def fit_year_trend(df, col):
    d = df.dropna(subset=[col]).copy()
    X = d[["year"]].values
    y = d[col].values
    return LinearRegression().fit(X, y)


def build_projection_lookups(df_hist):
    # city-month climatology
    clim_tavg = df_hist.groupby(["city","month"])["tavg_mean"].mean().rename("tavg_clim_proj").reset_index()
    clim_tmax = df_hist.groupby(["city","month"])["tmax_mean"].mean().rename("tmax_clim_proj").reset_index()
    clim_wspd = df_hist.groupby(["city","month"])["wspd_mean_filled"].median().rename("wspd_clim_proj").reset_index()
    anom_med  = df_hist.groupby(["city","month"])["tavg_anom"].median().rename("tavg_anom_med").reset_index()

    proj_lookup = clim_tavg.merge(clim_tmax, on=["city","month"]) \
                          .merge(clim_wspd, on=["city","month"]) \
                          .merge(anom_med,  on=["city","month"])

    # national surface temp monthly climatology + recent delta
    surf_clim = df_hist.groupby("month")["surface_temp_avg"].mean().rename("surf_clim").reset_index()
    last_year = df_hist["year"].max()
    recent = df_hist[df_hist["year"] >= last_year-1].groupby("month")["surface_temp_avg"].mean().rename("surf_recent").reset_index()
    surf_proj = surf_clim.merge(recent, on="month", how="left")
    surf_proj["surf_recent"] = surf_proj["surf_recent"].fillna(surf_proj["surf_clim"])
    surf_proj["surf_delta"] = surf_proj["surf_recent"] - surf_proj["surf_clim"]

    # yearly WB trends
    yearly = df_hist.groupby("year")[["pop_density","urban_pct"]].mean().reset_index()
    m_pop = fit_year_trend(yearly, "pop_density")
    m_urb = fit_year_trend(yearly, "urban_pct")

    return proj_lookup, surf_proj, m_pop, m_urb


def forecast_city(city, model, feature_cols, df_hist, proj_lookup, surf_proj, m_pop, m_urb, stats,
                  horizon_months=6, scenario=None):
    if scenario is None:
        scenario = {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}

    hist = df_hist[df_hist["city"] == city].sort_values(["year","month"]).copy()
    last = hist.iloc[-1]
    y, m = int(last["year"]), int(last["month"])
    heat_series = hist["heat_stress_index"].tolist()
    
    # Compute climatologies for humidity/NDVI features (for projections)
    clim_rh_range = None
    clim_ndvi = None
    if "rh_range" in feature_cols and "rh_range" in hist.columns:
        clim_rh_range = hist.groupby("month")["rh_range"].median()
    if "ndvi" in feature_cols and "ndvi" in hist.columns:
        clim_ndvi = hist.groupby("month")["ndvi"].median()

    out = []
    for _ in range(horizon_months):
        y, m = next_year_month(y, m)

        base = proj_lookup[(proj_lookup["city"] == city) & (proj_lookup["month"] == m)].iloc[0]
        srow = surf_proj[surf_proj["month"] == m].iloc[0]

        surface_temp_avg = float(srow["surf_clim"] + srow["surf_delta"])
        pop_density = float(m_pop.predict([[y]])[0]) * float(scenario["pop_delta_mult"])
        urban_pct   = float(m_urb.predict([[y]])[0]) + float(scenario["urban_delta_pct"])

        tavg_mean = float(base["tavg_clim_proj"] + base["tavg_anom_med"] + scenario["temp_delta_c"])
        tmax_mean = float(base["tmax_clim_proj"] + 0.8*base["tavg_anom_med"] + scenario["temp_delta_c"])
        wspd_mean_filled = float(base["wspd_clim_proj"])

        month_sin = np.sin(2*np.pi*m/12)
        month_cos = np.cos(2*np.pi*m/12)
        tavg_anom = tavg_mean - float(base["tavg_clim_proj"])

        heat_now = compute_heat_index({
            "tavg_mean": tavg_mean,
            "tmax_mean": tmax_mean,
            "surface_temp_avg": surface_temp_avg,
            "pop_density": pop_density,
            "urban_pct": urban_pct,
            "wspd_mean_filled": wspd_mean_filled,
        }, stats)
        heat_series.append(heat_now)

        prev = heat_series[:-1]
        def lag(L):  return heat_series[-1-L] if len(heat_series) > L else np.nan
        def rmean(w): return np.mean(prev[-w:]) if len(prev) >= w else np.nan
        def rstd(w):  return np.std(prev[-w:], ddof=1) if len(prev) >= w else np.nan

        feat = {
            "tavg_mean": tavg_mean, "tmax_mean": tmax_mean, "tavg_anom": tavg_anom,
            "surface_temp_avg": surface_temp_avg,
            "pop_density": pop_density, "urban_pct": urban_pct,
            "month_sin": month_sin, "month_cos": month_cos,
            "heat_lag_1": lag(1), "heat_lag_3": lag(3), "heat_lag_6": lag(6),
            "heat_roll_mean_3": rmean(3), "heat_roll_std_3": rstd(3),
            "heat_roll_mean_6": rmean(6), "heat_roll_std_6": rstd(6),
        }
        
        # Add humidity/NDVI features (use historical climatologies for projections)
        if "rh_range" in feature_cols:
            feat["rh_range"] = float(clim_rh_range.get(m, clim_rh_range.median())) if clim_rh_range is not None else np.nan
        if "ndvi" in feature_cols:
            feat["ndvi"] = float(clim_ndvi.get(m, clim_ndvi.median())) if clim_ndvi is not None else np.nan
        if "ndvi_missing" in feature_cols:
            feat["ndvi_missing"] = 0.0  # Assume data available in projections
        
        X_one = pd.DataFrame([feat])[feature_cols]
        proba = model.predict_proba(X_one)[0]
        pred = int(np.argmax(proba))

        out.append({
            "city": city, "year": y, "month": m,
            "pred_risk": pred,
            "p_low": float(proba[0]), "p_mod": float(proba[1]),
            "p_high": float(proba[2]), "p_extreme": float(proba[3]),
            "heat_stress_index_proj": float(heat_now),
            **scenario
        })
    return pd.DataFrame(out)


def main():
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    # Use FORECAST dataset (climate-only) for projections/history
    df_hist = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")

    # Load model: try DL first, fallback to HGB
    if (MODELS_DIR / "forecast_dl.pkl").exists():
        model = joblib.load(MODELS_DIR / "forecast_dl.pkl")
        model_type = "DL (HybridCNNLSTM)"
    else:
        model = joblib.load(MODELS_DIR / "forecast_hgb.pkl")
        model_type = "HGB"
    
    print(f"Using forecast model: {model_type}")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols_forecast.pkl")

    # stats for heat index computation
    stats = {
        "tavg_mean": (df_hist["tavg_mean"].mean(), df_hist["tavg_mean"].std()),
        "tmax_mean": (df_hist["tmax_mean"].mean(), df_hist["tmax_mean"].std()),
        "surface_temp_avg": (df_hist["surface_temp_avg"].mean(), df_hist["surface_temp_avg"].std()),
        "pop_density": (df_hist["pop_density"].mean(), df_hist["pop_density"].std()),
        "urban_pct": (df_hist["urban_pct"].mean(), df_hist["urban_pct"].std()),
        "wspd_mean_filled": (df_hist["wspd_mean_filled"].mean(), df_hist["wspd_mean_filled"].std()),
    }

    proj_lookup, surf_proj, m_pop, m_urb = build_projection_lookups(df_hist)

    cities = sorted(df_hist["city"].unique().tolist())

    def run(h, scenario, name):
        frames = []
        for c in cities:
            frames.append(forecast_city(c, model, feature_cols, df_hist, proj_lookup, surf_proj, m_pop, m_urb, stats, horizon_months=h, scenario=scenario))
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(FORECAST_DIR / f"forecast_{h}m_{name}.csv", index=False)

    # Original short-term forecasts
    run(6,  {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")
    run(12, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")
    run(24, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")

    # Medium-term forecasts (3-5 years)
    run(36, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")
    run(48, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")
    run(60, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")

    # Short-term scenarios (ensure dashboard has 12m/24m scenario files)
    run(12, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(24, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(12, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")
    run(24, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")

    # Medium-term scenarios
    run(36, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(48, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(60, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(36, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")
    run(48, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")
    run(60, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")

    # Extended forecasts to 2030 (72 months = 6 years from 2024)
    run(72, {"temp_delta_c":0.0, "urban_delta_pct":0.0, "pop_delta_mult":1.0}, "baseline")
    
    # Climate scenarios for extended horizon
    run(72, {"temp_delta_c":1.0, "urban_delta_pct":2.0,  "pop_delta_mult":1.05}, "plus1c")
    run(72, {"temp_delta_c":2.0, "urban_delta_pct":4.0,  "pop_delta_mult":1.10}, "plus2c")

    print("Saved forecasts to:", FORECAST_DIR)
    print(f"Generated forecasts with {model_type} model")
    print("Generated horizons: 6m, 12m, 24m, 36m, 48m, 60m, 72m (to 2030) + scenarios")


if __name__ == "__main__":
    main()
