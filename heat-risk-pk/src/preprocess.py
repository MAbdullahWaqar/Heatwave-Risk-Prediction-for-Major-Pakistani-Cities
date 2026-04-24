import pandas as pd
from .config import (
    MIN_MONTHS_PER_CITY,
    MIN_TAVG_NONNULL,
    HUMIDITY_WINSOR_LOW,
    HUMIDITY_WINSOR_HIGH,
    HUMIDITY_EXTREME_PRCP_Q,
)

def daily_to_monthly(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["time"] = pd.to_datetime(w["time"], errors="coerce")
    w = w.dropna(subset=["time", "city"])
    w["year"] = w["time"].dt.year
    w["month"] = w["time"].dt.month

    agg = {
        "tavg": ["mean", "max"],
        "tmin": ["mean", "min"],
        "tmax": ["mean", "max"],
        "prcp": ["sum", "mean"],
        "wspd": ["mean", "max"],
        "pres": ["mean"],
        "tsun": ["mean", "sum"],
    }
    monthly = w.groupby(["city", "year", "month"]).agg(agg)
    monthly.columns = ["_".join([a,b]) for a,b in monthly.columns]
    monthly = monthly.reset_index()
    monthly["n_days"] = w.groupby(["city", "year", "month"]).size().values
    return monthly

def filter_cities(monthly: pd.DataFrame) -> pd.DataFrame:
    coverage = monthly.groupby("city").agg(
        n_months=("year","size"),
        pct_tavg_nonnull=("tavg_mean", lambda s: s.notna().mean())
    )
    good = coverage[
        (coverage["n_months"] >= MIN_MONTHS_PER_CITY) &
        (coverage["pct_tavg_nonnull"] >= MIN_TAVG_NONNULL)
    ].index.tolist()
    return monthly[monthly["city"].isin(good)].copy()

def _winsorize_citywise(df: pd.DataFrame, col: str, q_low: float, q_high: float) -> pd.Series:
    low = df.groupby("city")[col].transform(lambda s: s.quantile(q_low))
    high = df.groupby("city")[col].transform(lambda s: s.quantile(q_high))
    return df[col].clip(lower=low, upper=high)

def humidity_daily_to_monthly(humidity: pd.DataFrame) -> pd.DataFrame:
    h = humidity.copy()
    h["time"] = pd.to_datetime(h["time"], errors="coerce")
    h = h.dropna(subset=["time", "city"])
    h["year"] = h["time"].dt.year
    h["month"] = h["time"].dt.month

    # Basic physical bounds before winsorization.
    h["rh_max"] = h["rh_max"].clip(lower=0, upper=100)
    h["rh_min"] = h["rh_min"].clip(lower=0, upper=100)
    h["rh_min"] = h[["rh_min", "rh_max"]].min(axis=1)

    for c in ["rh_max", "rh_min", "prcp", "et0"]:
        h[c] = _winsorize_citywise(h, c, HUMIDITY_WINSOR_LOW, HUMIDITY_WINSOR_HIGH)

    h["rh_range_daily"] = (h["rh_max"] - h["rh_min"]).clip(lower=0)
    h["rainy_day"] = (h["prcp"] > 0).astype(float)
    prcp_thr = h.groupby("city")["prcp"].transform(lambda s: s.quantile(HUMIDITY_EXTREME_PRCP_Q))
    h["extreme_rain_day"] = (h["prcp"] >= prcp_thr).astype(float)

    monthly = h.groupby(["city", "year", "month"], as_index=False).agg(
        rh_max_mean=("rh_max", "mean"),
        rh_min_mean=("rh_min", "mean"),
        rh_range=("rh_range_daily", "mean"),
        et0_mean=("et0", "mean"),
        humidity_prcp_sum=("prcp", "sum"),
        rainy_day_fraction=("rainy_day", "mean"),
        extreme_rain_days=("extreme_rain_day", "sum"),
        humidity_n_days=("prcp", "size"),
    )
    return monthly
