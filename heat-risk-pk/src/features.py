import numpy as np
import pandas as pd

def wb_to_long(df: pd.DataFrame, country_code="PAK", value_name="value") -> pd.DataFrame:
    d = df[df["Country Code"] == country_code].copy()
    year_cols = [c for c in d.columns if c.isdigit()]
    d = d[["Country Code"] + year_cols]
    out = d.melt(id_vars=["Country Code"], value_vars=year_cols, var_name="year", value_name=value_name)
    out["year"] = out["year"].astype(int)
    out = out.dropna(subset=[value_name])
    return out[["year", value_name]]

def surface_pk_monthly(surf: pd.DataFrame) -> pd.DataFrame:
    s = surf[surf["Code"] == "PAK"].copy()
    s["Day"] = pd.to_datetime(s["Day"], errors="coerce")
    s = s.dropna(subset=["Day"])
    s["year"] = s["Day"].dt.year
    s["month"] = s["Day"].dt.month
    s = s.rename(columns={"Average surface temperature": "surface_temp_avg"})
    return s[["year","month","surface_temp_avg"]].copy()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12)
    return out

def add_city_climatology(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    clim = out.groupby(["city","month"])["tavg_mean"].mean().rename("tavg_clim")
    out = out.merge(clim, on=["city","month"], how="left")
    out["tavg_anom"] = out["tavg_mean"] - out["tavg_clim"]
    return out


def ndvi_to_monthly(ndvi: pd.DataFrame) -> pd.DataFrame:
    out = ndvi.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time", "city", "ndvi"])
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    return out[["city", "year", "month", "ndvi"]]


def add_humidity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["rh_max_mean", "rh_min_mean", "rh_range", "et0_mean", "rainy_day_fraction"]:
        out[f"{c}_missing"] = out[c].isna().astype(int)

    # Fill with city-month climatology first, then city-level median, then global median.
    for c in ["rh_max_mean", "rh_min_mean", "rh_range", "et0_mean", "rainy_day_fraction", "extreme_rain_days", "humidity_prcp_sum"]:
        cm = out.groupby(["city", "month"])[c].transform("mean")
        city_med = out.groupby("city")[c].transform("median")
        out[c] = out[c].fillna(cm).fillna(city_med).fillna(out[c].median())

    rh_max_clim = out.groupby(["city", "month"])["rh_max_mean"].transform("mean")
    rh_min_clim = out.groupby(["city", "month"])["rh_min_mean"].transform("mean")
    et0_clim = out.groupby(["city", "month"])["et0_mean"].transform("mean")

    out["rh_max_anom"] = out["rh_max_mean"] - rh_max_clim
    out["rh_min_anom"] = out["rh_min_mean"] - rh_min_clim
    out["et0_anom"] = out["et0_mean"] - et0_clim

    out["rh_max_city_z"] = out.groupby("city")["rh_max_mean"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    out["rh_min_city_z"] = out.groupby("city")["rh_min_mean"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
    out["et0_city_z"] = out.groupby("city")["et0_mean"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))

    out["humid_temp_interaction_tavg"] = out["tavg_mean"] * out["rh_max_mean"]
    out["humid_temp_interaction_tmax"] = out["tmax_mean"] * out["rh_min_mean"]

    # Approximate apparent heat proxy from temperature and RH.
    out["apparent_heat_proxy"] = out["tavg_mean"] + 0.1 * (out["rh_max_mean"] - 50.0)

    return out
