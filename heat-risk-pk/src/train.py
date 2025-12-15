import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from .config import DATA_PROCESSED, MODELS_DIR
from .io import load_weather, load_worldbank_pop_density, load_worldbank_urban_pct, load_surface_temp
from .preprocess import daily_to_monthly, filter_cities
from .features import wb_to_long, surface_pk_monthly, add_time_features, add_city_climatology
from .targets import add_heat_index, add_risk_label
from .split import add_lags_rollings, temporal_split


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

    # minimal required cols for modeling to avoid broken rows
    df = df.dropna(subset=["tavg_mean","tmax_mean","pop_density","urban_pct","surface_temp_avg"]).copy()

    df = add_heat_index(df)
    df = add_risk_label(df)
    df = add_lags_rollings(df)

    return df


def get_feature_sets(df: pd.DataFrame):
    # Full feature set (Monitoring) includes risk_lags
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

    # Climate-only feature set removes risk_lags
    feature_cols_climate = [c for c in feature_cols_full if not c.startswith("risk_lag_")]

    # Drop rows missing any required features
    df_full = df.dropna(subset=feature_cols_full + ["risk_label"]).copy()
    df_clim = df.dropna(subset=feature_cols_climate + ["risk_label"]).copy()

    return feature_cols_full, feature_cols_climate, df_full, df_clim


def train_models(df_full: pd.DataFrame, df_clim: pd.DataFrame, feature_cols_full, feature_cols_climate):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Splits
    train_f, val_f, test_f = temporal_split(df_full)
    train_c, val_c, test_c = temporal_split(df_clim)

    X_train_f, y_train_f = train_f[feature_cols_full], train_f["risk_label"]
    X_val_f,   y_val_f   = val_f[feature_cols_full],   val_f["risk_label"]
    X_test_f,  y_test_f  = test_f[feature_cols_full],  test_f["risk_label"]

    X_train_c, y_train_c = train_c[feature_cols_climate], train_c["risk_label"]
    X_val_c,   y_val_c   = val_c[feature_cols_climate],   val_c["risk_label"]
    X_test_c,  y_test_c  = test_c[feature_cols_climate],  test_c["risk_label"]

    # -------------------------
    # Monitoring model (uses risk_lags) - Logistic Regression
    # -------------------------
    mon_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    mon_model.fit(X_train_f, y_train_f)

    mon_pred = mon_model.predict(X_test_f)
    mon_f1 = f1_score(y_test_f, mon_pred, average="macro")
    mon_acc = accuracy_score(y_test_f, mon_pred)

    # -------------------------
    # Forecast model (climate-only) - Gradient Boosting
    # -------------------------
    fc_model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
    )
    fc_model.fit(X_train_c, y_train_c)

    fc_pred = fc_model.predict(X_test_c)
    fc_f1 = f1_score(y_test_c, fc_pred, average="macro")
    fc_acc = accuracy_score(y_test_c, fc_pred)

    # Optional explainable model for SHAP (climate-only)
    rf_explain = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=15,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf_explain.fit(X_train_c, y_train_c)

    # Save models + columns
    joblib.dump(mon_model, MODELS_DIR / "monitoring_logreg.pkl")
    joblib.dump(fc_model,  MODELS_DIR / "forecast_hgb.pkl")
    joblib.dump(rf_explain, MODELS_DIR / "explain_rf.pkl")

    joblib.dump(feature_cols_full,    MODELS_DIR / "feature_cols_monitoring.pkl")
    joblib.dump(feature_cols_climate, MODELS_DIR / "feature_cols_forecast.pkl")

    # Save processed datasets (optional but useful)
    df_full.to_csv(DATA_PROCESSED / "df_model_monitoring.csv", index=False)
    df_clim.to_csv(DATA_PROCESSED / "df_model_forecast.csv", index=False)

    metrics = {
        "monitoring_logreg": {"macro_f1": float(mon_f1), "accuracy": float(mon_acc)},
        "forecast_hgb": {"macro_f1": float(fc_f1), "accuracy": float(fc_acc)},
        "notes": {
            "monitoring_model": "Uses risk_lag_* features; best for monitoring/alerts",
            "forecast_model": "Climate-only; best for forward scenario forecasting"
        }
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    df = build_dataset()
    feature_cols_full, feature_cols_climate, df_full, df_clim = get_feature_sets(df)
    metrics = train_models(df_full, df_clim, feature_cols_full, feature_cols_climate)
    print("Saved models + metrics to:", MODELS_DIR)
    print(metrics)


if __name__ == "__main__":
    main()
