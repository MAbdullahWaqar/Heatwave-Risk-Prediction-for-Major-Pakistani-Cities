import json
import joblib

import pandas as pd

from .config import DATA_PROCESSED, MODELS_DIR
from .io import load_weather, load_worldbank_pop_density, load_worldbank_urban_pct, load_surface_temp
from .preprocess import daily_to_monthly, filter_cities
from .features import wb_to_long, surface_pk_monthly, add_time_features, add_city_climatology
from .targets import add_heat_index, add_risk_label
from .split import add_lags_rollings


def build_dataset() -> pd.DataFrame:
    weather = load_weather()
    pop_raw = load_worldbank_pop_density()
    urb_raw = load_worldbank_urban_pct()
    surf_raw = load_surface_temp()

    monthly = daily_to_monthly(weather)
    monthly = filter_cities(monthly)

    pop_pak = wb_to_long(pop_raw, "PAK", "pop_density")
    urb_pak = wb_to_long(urb_raw, "PAK", "urban_pct")
    surf_pak = surface_pk_monthly(surf_raw)

    df = monthly.merge(pop_pak, on="year", how="left") \
                .merge(urb_pak, on="year", how="left") \
                .merge(surf_pak, on=["year", "month"], how="left")

    df = add_time_features(df)
    df = add_city_climatology(df)

    df = df.dropna(subset=["tavg_mean", "tmax_mean", "pop_density", "urban_pct", "surface_temp_avg"]).copy()

    df = add_heat_index(df)
    df = add_risk_label(df)
    df = add_lags_rollings(df)

    return df


def get_feature_sets(df: pd.DataFrame):
    feature_cols_full = [
        "tavg_mean", "tmax_mean", "tavg_anom",
        "surface_temp_avg",
        "pop_density", "urban_pct",
        "month_sin", "month_cos",
        "heat_lag_1", "heat_lag_3", "heat_lag_6",
        "risk_lag_1", "risk_lag_3", "risk_lag_6",
        "heat_roll_mean_3", "heat_roll_std_3",
        "heat_roll_mean_6", "heat_roll_std_6",
    ]

    feature_cols_climate = [c for c in feature_cols_full if not c.startswith("risk_lag_")]

    df_full = df.dropna(subset=feature_cols_full + ["risk_label"]).copy()
    df_clim = df.dropna(subset=feature_cols_climate + ["risk_label"]).copy()

    return feature_cols_full, feature_cols_climate, df_full, df_clim


def save_processed_for_gru(
    df_full: pd.DataFrame,
    df_clim: pd.DataFrame,
    feature_cols_full: list,
    feature_cols_climate: list,
) -> dict:
    """Write processed tables and feature lists for GRU training (notebook) and forecasting. No sklearn risk models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    joblib.dump(feature_cols_full, MODELS_DIR / "feature_cols_monitoring.pkl")
    joblib.dump(feature_cols_climate, MODELS_DIR / "feature_cols_forecast.pkl")

    df_full.to_csv(DATA_PROCESSED / "df_model_monitoring.csv", index=False)
    df_clim.to_csv(DATA_PROCESSED / "df_model_forecast.csv", index=False)

    metrics = {
        "pipeline": "gru_only",
        "notes": {
            "forecast": "Train and export GRU via notebooks/deep_learning_model_selection.ipynb → models/gru_attn_best.pkl",
            "data": "df_model_forecast.csv is the GRU training table; feature_cols_forecast.pkl lists columns.",
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    df = build_dataset()
    feature_cols_full, feature_cols_climate, df_full, df_clim = get_feature_sets(df)
    metrics = save_processed_for_gru(df_full, df_clim, feature_cols_full, feature_cols_climate)
    print("Saved processed data + feature lists to:", DATA_PROCESSED, "and", MODELS_DIR)
    print(metrics)


if __name__ == "__main__":
    main()
