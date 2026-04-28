"""
GRU risk forecast (recursive, bidirectional GRU + attention) using the same scenario structure as the legacy HGB path.
Reads `models/<SEQUENCE_CHECKPOINT_NAME>` (default `gru_attn_best.pkl`; torch checkpoint from the DL notebook).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED, FORECAST_DIR, MODELS_DIR, SEQUENCE_CHECKPOINT_NAME, TRAIN_END_YEAR
from .forecast import (
    build_projection_lookups,
    compute_heat_index,
    next_year_month,
)
from .lstm_risk_model import load_lstm_checkpoint
from .merge_dl_features import merge_auxiliary_features


def _col_z(x, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return (float(x) - float(mean)) / float(std)


def _build_feature_row(
    city: str,
    year: int,
    month: int,
    scenario: dict,
    base: pd.Series,
    srow: pd.Series,
    m_pop,
    m_urb,
    stats_heat: dict,
    stats_col: dict,
    feature_cols: list[str],
    heat_series_with_new: list[float],
    clim_extras: pd.DataFrame,
) -> dict:
    """heat_series_with_new includes the new month heat at the end."""
    surface_temp_avg = float(srow["surf_clim"] + srow["surf_delta"])
    pop_density = float(m_pop.predict([[year]])[0]) * float(scenario["pop_delta_mult"])
    urban_pct = float(m_urb.predict([[year]])[0]) + float(scenario["urban_delta_pct"])
    tavg_mean = float(base["tavg_clim_proj"] + base["tavg_anom_med"] + scenario["temp_delta_c"])
    tmax_mean = float(base["tmax_clim_proj"] + 0.8 * base["tavg_anom_med"] + scenario["temp_delta_c"])
    wspd_mean_filled = float(base["wspd_clim_proj"])
    tavg_clim = float(base["tavg_clim_proj"])
    tavg_anom = tavg_mean - tavg_clim
    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    heat_now = float(heat_series_with_new[-1])

    prev = heat_series_with_new[:-1]

    def lag(L):
        return heat_series_with_new[-1 - L] if len(heat_series_with_new) > L else np.nan

    def rmean(w):
        return float(np.mean(prev[-w:])) if len(prev) >= w else np.nan

    def rstd(w):
        return float(np.std(prev[-w:], ddof=1)) if len(prev) >= w else np.nan

    row = {
        "tavg_mean": tavg_mean,
        "tmax_mean": tmax_mean,
        "tmin_mean": tavg_mean - 4.0,
        "tmin_min": tavg_mean - 8.0,
        "tavg_max": tmax_mean,
        "tmax_max": tmax_mean + 2.0,
        "prcp_sum": 0.0,
        "prcp_mean": 0.0,
        "wspd_mean": wspd_mean_filled,
        "wspd_max": wspd_mean_filled * 1.1,
        "pres_mean": 1005.0,
        "tsun_mean": 200.0,
        "tsun_sum": 6000.0,
        "n_days": 30.0,
        "pop_density": pop_density,
        "urban_pct": urban_pct,
        "surface_temp_avg": surface_temp_avg,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "tavg_clim": tavg_clim,
        "tavg_anom": tavg_anom,
        "wspd_mean_filled": wspd_mean_filled,
        "heat_stress_index": heat_now,
        "heat_lag_1": lag(1),
        "heat_lag_3": lag(3),
        "heat_lag_6": lag(6),
        "heat_roll_mean_3": rmean(3),
        "heat_roll_std_3": rstd(3),
        "heat_roll_mean_6": rmean(6),
        "heat_roll_std_6": rstd(6),
    }
    row["tavg_z"] = _col_z(tavg_mean, *stats_col["tavg_mean"])
    row["tmax_z"] = _col_z(tmax_mean, *stats_col["tmax_mean"])
    row["surf_z"] = _col_z(surface_temp_avg, *stats_col["surface_temp_avg"])
    row["pop_z"] = _col_z(pop_density, *stats_col["pop_density"])
    row["urb_z"] = _col_z(urban_pct, *stats_col["urban_pct"])
    row["wind_z"] = _col_z(wspd_mean_filled, *stats_col["wspd_mean_filled"])

    exm = clim_extras[(clim_extras["city"] == city) & (clim_extras["month"] == month)]
    if len(exm) > 0:
        r0 = exm.iloc[0]
        for c in feature_cols:
            if c.startswith(("hum_", "ndvi_", "wm_")) and c in r0.index and c != "city":
                v = r0[c]
                if not pd.isna(v):
                    row[c] = float(v)

    for c in feature_cols:
        if c not in row or pd.isna(row.get(c, np.nan)) or (isinstance(row.get(c), float) and np.isnan(row.get(c, 0.0))):
            m = stats_col.get(c, (0.0, 1.0))[0]
            row[c] = float(m)

    return {k: float(row[k]) for k in feature_cols}


def forecast_city_lstm(
    city: str,
    model: torch.nn.Module,
    feature_cols: list[str],
    city_to_idx: dict[str, int],
    scaler: StandardScaler,
    device: torch.device,
    stats_heat: dict,
    stats_col: dict,
    df_m: pd.DataFrame,
    proj_lookup,
    surf_proj,
    m_pop,
    m_urb,
    clim_extras: pd.DataFrame,
    seq_len: int,
    horizon_months: int = 6,
    scenario: dict | None = None,
) -> pd.DataFrame:
    if scenario is None:
        scenario = {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}

    mhist = df_m[df_m["city"] == city].sort_values(["year", "month"])
    if len(mhist) < seq_len + 1:
        raise ValueError(f"Not enough history for {city} (need at least {seq_len + 1} rows)")

    heat_series = mhist["heat_stress_index"].astype(float).tolist()
    y0, m0 = int(mhist.iloc[-1]["year"]), int(mhist.iloc[-1]["month"])
    cidx = int(city_to_idx[city])
    out: list[dict] = []
    win_rows: list[np.ndarray] = [mhist[feature_cols].values.astype(np.float32)[i] for i in range(-seq_len, 0)]

    for _ in range(horizon_months):
        y, m = next_year_month(y0, m0)
        y0, m0 = y, m
        base = proj_lookup[(proj_lookup["city"] == city) & (proj_lookup["month"] == m)].iloc[0]
        srow = surf_proj[surf_proj["month"] == m].iloc[0]
        tavg_i = float(base["tavg_clim_proj"] + base["tavg_anom_med"] + scenario["temp_delta_c"])
        tmax_i = float(base["tmax_clim_proj"] + 0.8 * base["tavg_anom_med"] + scenario["temp_delta_c"])
        srf = float(srow["surf_clim"] + srow["surf_delta"])
        popd = float(m_pop.predict([[y]])[0]) * float(scenario["pop_delta_mult"])
        urbd = float(m_urb.predict([[y]])[0]) + float(scenario["urban_delta_pct"])
        wspd = float(base["wspd_clim_proj"])
        h_add = float(
            compute_heat_index(
                {
                    "tavg_mean": tavg_i,
                    "tmax_mean": tmax_i,
                    "surface_temp_avg": srf,
                    "pop_density": popd,
                    "urban_pct": urbd,
                    "wspd_mean_filled": wspd,
                },
                stats_heat,
            )
        )
        heat_series = heat_series + [h_add]
        rowd = _build_feature_row(
            city, y, m, scenario, base, srow, m_pop, m_urb, stats_heat, stats_col, feature_cols, heat_series, clim_extras
        )
        row_vec = np.array([rowd[c] for c in feature_cols], dtype=np.float32)
        win_rows = win_rows[1:] + [row_vec]
        X = np.stack(win_rows, axis=0)
        Xs = scaler.transform(X)
        with torch.no_grad():
            xb = torch.tensor(Xs.reshape(1, seq_len, -1), dtype=torch.float32, device=device)
            cb = torch.tensor([cidx], dtype=torch.long, device=device)
            logits = model(xb, cb)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(proba))
        out.append(
            {
                "city": city,
                "year": y,
                "month": m,
                "pred_risk": pred,
                "p_low": float(proba[0]),
                "p_mod": float(proba[1]),
                "p_high": float(proba[2]),
                "p_extreme": float(proba[3]),
                "heat_stress_index_proj": h_add,
                **scenario,
            }
        )
    return pd.DataFrame(out)


def run_lstm_main(checkpoint_name: str | None = None) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    ck_name = checkpoint_name or SEQUENCE_CHECKPOINT_NAME
    ck = MODELS_DIR / ck_name
    if not ck.is_file():
        raise FileNotFoundError(f"Missing GRU checkpoint: {ck}")

    model, payload = load_lstm_checkpoint(ck, device=torch.device("cpu"))
    feature_cols: list = payload["feature_cols"]
    city_to_idx: dict = payload["city_to_idx"]
    cfg: dict = payload.get("config", {})
    seq_len = int(cfg.get("seq_len", 12))

    df_base = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    df_m = merge_auxiliary_features(df_base)

    for c in feature_cols:
        if c not in df_m.columns:
            df_m[c] = np.nan
    tr = df_m[df_m["year"] <= TRAIN_END_YEAR]
    for c in feature_cols:
        med = tr[c].median()
        if pd.isna(med):
            med = 0.0
        df_m[c] = df_m[c].fillna(med)
        df_m[c] = df_m[c].fillna(0.0)
    df_m[feature_cols] = (
        df_m[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    scaler = StandardScaler()
    X_tr = np.asarray(tr[feature_cols].values, dtype=np.float64)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(X_tr)

    ex_cols = [c for c in feature_cols if c.startswith(("hum_", "ndvi_", "wm_")) and c in df_m.columns]
    if ex_cols:
        clim_extras = df_m.groupby(["city", "month"])[ex_cols].median().reset_index()
    else:
        clim_extras = df_m[["city", "month"]].drop_duplicates().reset_index(drop=True)

    stats_heat = {
        "tavg_mean": (df_m["tavg_mean"].mean(), max(df_m["tavg_mean"].std(), 1e-6)),
        "tmax_mean": (df_m["tmax_mean"].mean(), max(df_m["tmax_mean"].std(), 1e-6)),
        "surface_temp_avg": (df_m["surface_temp_avg"].mean(), max(df_m["surface_temp_avg"].std(), 1e-6)),
        "pop_density": (df_m["pop_density"].mean(), max(df_m["pop_density"].std(), 1e-6)),
        "urban_pct": (df_m["urban_pct"].mean(), max(df_m["urban_pct"].std(), 1e-6)),
        "wspd_mean_filled": (df_m["wspd_mean_filled"].mean(), max(df_m["wspd_mean_filled"].std(), 1e-6)),
    }
    def _mean_std_1d(a) -> tuple[float, float]:
        v = np.asarray(a, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 1e-6
        m = float(np.mean(v))
        s = float(np.std(v, ddof=0))
        return m, max(s, 1e-6)

    stats_col = {c: _mean_std_1d(tr[c].to_numpy()) for c in feature_cols}

    proj_lookup, surf_proj, m_pop, m_urb = build_projection_lookups(df_base)
    cities = sorted(df_m["city"].unique().tolist())
    dev = torch.device("cpu")
    model.to(dev)

    def run_h(h, sc, name):
        frames = []
        for c in cities:
            try:
                frames.append(
                    forecast_city_lstm(
                        c,
                        model,
                        feature_cols,
                        city_to_idx,
                        scaler,
                        dev,
                        stats_heat,
                        stats_col,
                        df_m,
                        proj_lookup,
                        surf_proj,
                        m_pop,
                        m_urb,
                        clim_extras,
                        seq_len,
                        h,
                        sc,
                    )
                )
            except Exception as e:
                print(f"Skip {c}: {e}")
        if not frames:
            raise RuntimeError("No GRU forecast frames")
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, "forecast_model", str(payload.get("model_name", "GRU_Attn")))
        out.insert(1, "forecast_checkpoint", ck.name)
        out.to_csv(FORECAST_DIR / f"forecast_{h}m_{name}.csv", index=False)

    run_h(6, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(12, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(24, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(36, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(48, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(60, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(12, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(24, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(12, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")
    run_h(24, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")
    run_h(36, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(48, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(60, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(36, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")
    run_h(48, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")
    run_h(60, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")
    run_h(72, {"temp_delta_c": 0.0, "urban_delta_pct": 0.0, "pop_delta_mult": 1.0}, "baseline")
    run_h(72, {"temp_delta_c": 1.0, "urban_delta_pct": 2.0, "pop_delta_mult": 1.05}, "plus1c")
    run_h(72, {"temp_delta_c": 2.0, "urban_delta_pct": 4.0, "pop_delta_mult": 1.10}, "plus2c")

    print("Saved GRU forecasts to:", FORECAST_DIR)
    print("Model:", payload.get("model_name"), "checkpoint:", ck.name)


if __name__ == "__main__":
    run_lstm_main()
