import pandas as pd
from .config import MIN_MONTHS_PER_CITY, MIN_TAVG_NONNULL

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
