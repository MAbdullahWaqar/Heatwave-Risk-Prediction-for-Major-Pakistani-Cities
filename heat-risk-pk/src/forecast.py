import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import DATA_PROCESSED, FORECAST_DIR


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
    pop_z = z(row["pop_density"], *stats["pop_density"])
    urb_z = z(row["urban_pct"], *stats["urban_pct"])
    wind_z = z(row["wspd_mean_filled"], *stats["wspd_mean_filled"])
    return (
        0.40 * tavg_z
        + 0.30 * tmax_z
        + 0.15 * surf_z
        + 0.10 * pop_z
        + 0.10 * urb_z
        - 0.05 * wind_z
    )


def fit_year_trend(df, col):
    d = df.dropna(subset=[col]).copy()
    X = d[["year"]].values
    y = d[col].values
    return LinearRegression().fit(X, y)


def build_projection_lookups(df_hist):
    clim_tavg = df_hist.groupby(["city", "month"])["tavg_mean"].mean().rename("tavg_clim_proj").reset_index()
    clim_tmax = df_hist.groupby(["city", "month"])["tmax_mean"].mean().rename("tmax_clim_proj").reset_index()
    clim_wspd = df_hist.groupby(["city", "month"])["wspd_mean_filled"].median().rename("wspd_clim_proj").reset_index()
    anom_med = df_hist.groupby(["city", "month"])["tavg_anom"].median().rename("tavg_anom_med").reset_index()

    proj_lookup = (
        clim_tavg.merge(clim_tmax, on=["city", "month"])
        .merge(clim_wspd, on=["city", "month"])
        .merge(anom_med, on=["city", "month"])
    )

    surf_clim = df_hist.groupby("month")["surface_temp_avg"].mean().rename("surf_clim").reset_index()
    last_year = df_hist["year"].max()
    recent = (
        df_hist[df_hist["year"] >= last_year - 1]
        .groupby("month")["surface_temp_avg"]
        .mean()
        .rename("surf_recent")
        .reset_index()
    )
    surf_proj = surf_clim.merge(recent, on="month", how="left")
    surf_proj["surf_recent"] = surf_proj["surf_recent"].fillna(surf_proj["surf_clim"])
    surf_proj["surf_delta"] = surf_proj["surf_recent"] - surf_proj["surf_clim"]

    yearly = df_hist.groupby("year")[["pop_density", "urban_pct"]].mean().reset_index()
    m_pop = fit_year_trend(yearly, "pop_density")
    m_urb = fit_year_trend(yearly, "urban_pct")

    return proj_lookup, surf_proj, m_pop, m_urb


def main():
    """Generate forecasts with the GRU + Attention checkpoint (see `SEQUENCE_CHECKPOINT_NAME` in `config.py`)."""
    from .forecast_lstm import run_lstm_main

    run_lstm_main()


if __name__ == "__main__":
    main()
