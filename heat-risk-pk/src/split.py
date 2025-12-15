import pandas as pd
from .config import TRAIN_END_YEAR, VAL_END_YEAR

def add_lags_rollings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["city","year","month"]).copy()

    for lag in [1,3,6]:
        out[f"heat_lag_{lag}"] = out.groupby("city")["heat_stress_index"].shift(lag)
        out[f"risk_lag_{lag}"] = out.groupby("city")["risk_label"].shift(lag)

    for w in [3,6]:
        out[f"heat_roll_mean_{w}"] = out.groupby("city")["heat_stress_index"].shift(1).rolling(w).mean().values
        out[f"heat_roll_std_{w}"]  = out.groupby("city")["heat_stress_index"].shift(1).rolling(w).std().values

    return out

def temporal_split(df_model: pd.DataFrame):
    train = df_model[df_model["year"] <= TRAIN_END_YEAR]
    val   = df_model[(df_model["year"] > TRAIN_END_YEAR) & (df_model["year"] <= VAL_END_YEAR)]
    test  = df_model[df_model["year"] > VAL_END_YEAR]
    return train, val, test
