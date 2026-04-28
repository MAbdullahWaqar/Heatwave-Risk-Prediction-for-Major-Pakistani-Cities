"""Merge humidity, NDVI, and optional merged-weather columns into the base forecast table (matches DL notebook)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW, DATA_PROCESSED


def merge_auxiliary_features(
    df: pd.DataFrame,
    humidity_path: Path | None = None,
    ndvi_path: Path | None = None,
    merged_weather_path: Path | None = None,
) -> pd.DataFrame:
    """
    `df` must include city, year, month and base numeric columns.
    """
    out = df.copy()
    out["city"] = out["city"].astype(str).str.strip()

    humidity_path = humidity_path or (DATA_RAW / "pakistan_humidity_daily.csv")
    if humidity_path.is_file():
        hum = pd.read_csv(humidity_path)
        hum["time"] = pd.to_datetime(hum["time"], errors="coerce")
        hum = hum.dropna(subset=["time", "city"])
        hum["city"] = hum["city"].astype(str).str.strip()
        hum["year"] = hum["time"].dt.year
        hum["month"] = hum["time"].dt.month
        hum_cols = [c for c in ["rh_avg", "rh_max", "rh_min", "prcp", "et0"] if c in hum.columns]
        if hum_cols:
            hum_agg = hum.groupby(["city", "year", "month"], as_index=False)[hum_cols].mean()
            hum_agg = hum_agg.rename(columns={c: f"hum_{c}" for c in hum_cols})
            out = out.merge(
                hum_agg,
                how="left",
                left_on=["city", "year", "month"],
                right_on=["city", "year", "month"],
            )
            if "city_x" in out.columns:
                out = out.drop(columns=[c for c in out.columns if c.endswith("_x") and "city" in c])

    ndvi_path = ndvi_path or (DATA_RAW / "pakistan_ndvi_monthly.csv")
    if ndvi_path.is_file():
        ndvi = pd.read_csv(ndvi_path)
        ndvi["time"] = pd.to_datetime(ndvi["time"], errors="coerce")
        ndvi = ndvi.dropna(subset=["time", "city"])
        ndvi["city"] = ndvi["city"].astype(str).str.strip()
        ndvi["year"] = ndvi["time"].dt.year
        ndvi["month"] = ndvi["time"].dt.month
        ndvi_m = ndvi.groupby(["city", "year", "month"], as_index=False)["ndvi"].mean()
        ndvi_m = ndvi_m.rename(columns={"ndvi": "ndvi_monthly"})
        out = out.merge(
            ndvi_m,
            how="left",
            on=["city", "year", "month"],
        )

    merged_weather_path = merged_weather_path or (DATA_PROCESSED / "pakistan_weather_merged_scaled.csv")
    if merged_weather_path.is_file():
        wm = pd.read_csv(merged_weather_path)
        if {"time", "city"}.issubset(wm.columns):
            wm["time"] = pd.to_datetime(wm["time"], errors="coerce")
            wm = wm.dropna(subset=["time", "city"])
            wm["city"] = wm["city"].astype(str).str.strip()
            wm["year"] = wm["time"].dt.year
            wm["month"] = wm["time"].dt.month
            wm_candidate = [
                c
                for c in ["rh_avg_scaled", "prcp_scaled", "et0_scaled", "rh_avg", "prcp", "et0"]
                if c in wm.columns
            ]
            if wm_candidate:
                wm_m = wm.groupby(["city", "year", "month"], as_index=False)[wm_candidate].mean()
                wm_m = wm_m.rename(columns={c: f"wm_{c}" for c in wm_candidate})
                out = out.merge(
                    wm_m,
                    how="left",
                    on=["city", "year", "month"],
                )

    return out
