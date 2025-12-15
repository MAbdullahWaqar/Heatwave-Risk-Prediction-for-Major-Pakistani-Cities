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
