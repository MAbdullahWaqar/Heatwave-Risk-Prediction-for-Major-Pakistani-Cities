import numpy as np
import pandas as pd
from .config import P50, P75, P90

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()

def add_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["tavg_z"] = zscore(out["tavg_mean"])
    out["tmax_z"] = zscore(out["tmax_mean"])
    out["surf_z"] = zscore(out["surface_temp_avg"])
    out["pop_z"]  = zscore(out["pop_density"])
    out["urb_z"]  = zscore(out["urban_pct"])

    out["wspd_mean_filled"] = out["wspd_mean"].fillna(out["wspd_mean"].median())
    out["wind_z"] = zscore(out["wspd_mean_filled"])

    out["heat_stress_index"] = (
        0.40*out["tavg_z"] +
        0.30*out["tmax_z"] +
        0.15*out["surf_z"] +
        0.10*out["pop_z"] +
        0.10*out["urb_z"] -
        0.05*out["wind_z"]
    )
    return out

def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p50 = out["heat_stress_index"].quantile(P50)
    p75 = out["heat_stress_index"].quantile(P75)
    p90 = out["heat_stress_index"].quantile(P90)

    def lab(x):
        if x <= p50: return 0
        if x <= p75: return 1
        if x <= p90: return 2
        return 3

    out["risk_label"] = out["heat_stress_index"].apply(lab)
    return out
