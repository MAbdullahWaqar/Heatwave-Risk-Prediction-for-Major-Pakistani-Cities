import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from .config import DATA_PROCESSED, MODELS_DIR
from .io import (
    load_weather,
    load_humidity,
    load_ndvi,
    load_worldbank_pop_density,
    load_worldbank_urban_pct,
    load_surface_temp,
)
from .preprocess import daily_to_monthly, humidity_daily_to_monthly, filter_cities
from .features import (
    wb_to_long,
    surface_pk_monthly,
    ndvi_to_monthly,
    add_time_features,
    add_city_climatology,
    add_humidity_features,
)
from .targets import add_heat_index, add_risk_label
from .split import add_lags_rollings, temporal_split
from .dl_models import DLForecastModel


def build_dataset() -> pd.DataFrame:
    weather = load_weather()
    humidity = load_humidity()
    ndvi = load_ndvi()
    pop_raw = load_worldbank_pop_density()
    urb_raw = load_worldbank_urban_pct()
    surf_raw = load_surface_temp()

    monthly = daily_to_monthly(weather)
    humidity_monthly = humidity_daily_to_monthly(humidity)
    ndvi_monthly = ndvi_to_monthly(ndvi)
    monthly = filter_cities(monthly)

    pop_pak = wb_to_long(pop_raw, "PAK", "pop_density")
    urb_pak = wb_to_long(urb_raw, "PAK", "urban_pct")
    surf_pak = surface_pk_monthly(surf_raw)

    df = (
        monthly.merge(pop_pak, on="year", how="left")
        .merge(urb_pak, on="year", how="left")
        .merge(surf_pak, on=["year", "month"], how="left")
        .merge(humidity_monthly, on=["city", "year", "month"], how="left")
        .merge(ndvi_monthly, on=["city", "year", "month"], how="left")
    )

    df = add_time_features(df)
    df = add_city_climatology(df)
    df = add_humidity_features(df)

    df["ndvi_missing"] = df["ndvi"].isna().astype(int)
    ndvi_cm = df.groupby(["city", "month"])["ndvi"].transform("mean")
    ndvi_city_med = df.groupby("city")["ndvi"].transform("median")
    df["ndvi"] = df["ndvi"].fillna(ndvi_cm).fillna(ndvi_city_med).fillna(df["ndvi"].median())

    # Minimal required cols for modeling to avoid broken rows.
    df = df.dropna(subset=["tavg_mean", "tmax_mean", "pop_density", "urban_pct", "surface_temp_avg"]).copy()

    df = add_heat_index(df)
    df = add_risk_label(df)
    df = add_lags_rollings(df)

    return df


def get_ablation_feature_groups() -> dict:
    return {
        "humidity_anomaly": ["rh_max_anom", "rh_min_anom", "rh_range", "et0_anom"],
        "humidity_interactions": [
            "humid_temp_interaction_tavg",
            "humid_temp_interaction_tmax",
            "apparent_heat_proxy",
        ],
        "humidity_lagged": [
            "rh_max_lag_1",
            "rh_max_lag_3",
            "rh_max_lag_6",
            "rh_max_roll_mean_3",
            "rh_max_roll_mean_6",
            "et0_roll_mean_3",
            "et0_roll_mean_6",
        ],
        "humidity_compact": [
            "rh_range",
            "ndvi",
            "ndvi_missing",
        ],
    }


def get_feature_sets(df: pd.DataFrame):
    base_common = [
        "tavg_mean",
        "tmax_mean",
        "tavg_anom",
        "surface_temp_avg",
        "pop_density",
        "urban_pct",
        "month_sin",
        "month_cos",
        "heat_lag_1",
        "heat_lag_3",
        "heat_lag_6",
        "heat_roll_mean_3",
        "heat_roll_std_3",
        "heat_roll_mean_6",
        "heat_roll_std_6",
    ]
    humidity_compact = get_ablation_feature_groups()["humidity_compact"]

    feature_cols_full = base_common + ["risk_lag_1", "risk_lag_3", "risk_lag_6"] + humidity_compact
    feature_cols_climate = base_common + humidity_compact

    df_full = df.dropna(subset=feature_cols_full + ["risk_label"]).copy()
    df_clim = df.dropna(subset=feature_cols_climate + ["risk_label"]).copy()

    return feature_cols_full, feature_cols_climate, df_full, df_clim


def train_models(df_full: pd.DataFrame, df_clim: pd.DataFrame, feature_cols_full, feature_cols_climate, use_dl: bool = True):
    """Train monitoring and forecast models.
    
    Args:
        df_full: Full dataset with monitoring features (includes risk lags)
        df_clim: Climate dataset for forecast (no risk lags, for true forecasting)
        feature_cols_full: Feature names for monitoring
        feature_cols_climate: Feature names for forecast
        use_dl: If True, use DL model for forecast; else use HGB
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train_f, val_f, test_f = temporal_split(df_full)
    train_c, val_c, test_c = temporal_split(df_clim)

    X_train_f, y_train_f = train_f[feature_cols_full], train_f["risk_label"]
    X_val_f, y_val_f = val_f[feature_cols_full], val_f["risk_label"]
    X_test_f, y_test_f = test_f[feature_cols_full], test_f["risk_label"]

    X_train_c, y_train_c = train_c[feature_cols_climate], train_c["risk_label"]
    X_val_c, y_val_c = val_c[feature_cols_climate], val_c["risk_label"]
    X_test_c, y_test_c = test_c[feature_cols_climate], test_c["risk_label"]

    # === Monitoring Model (LogReg with risk lags) ===
    mon_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    mon_model.fit(X_train_f, y_train_f)
    mon_pred = mon_model.predict(X_test_f)
    mon_f1 = f1_score(y_test_f, mon_pred, average="macro")
    mon_acc = accuracy_score(y_test_f, mon_pred)

    # === Forecast Model (DL or HGB) ===
    if use_dl:
        print("Training DL forecast model (HybridCNNLSTM)...")
        fc_model = DLForecastModel(
            input_size=len(feature_cols_climate),
            num_classes=4,
            hidden_size=64,
            learning_rate=0.001,
            device="cpu"
        )
        
        # Reshape data for sequence model (seq_len=1 since we're not windowing, just using features as channels)
        X_train_c_seq = np.expand_dims(X_train_c.values, axis=1)  # (n, 1, n_features)
        X_val_c_seq = np.expand_dims(X_val_c.values, axis=1)
        X_test_c_seq = np.expand_dims(X_test_c.values, axis=1)
        
        fc_model.fit(
            X_train_c_seq, y_train_c.values,
            X_val_c_seq, y_val_c.values,
            epochs=60,
            batch_size=16,
            patience=10
        )
        
        eval_metrics = fc_model.evaluate(X_test_c_seq, y_test_c.values)
        fc_f1 = eval_metrics["macro_f1"]
        fc_acc = eval_metrics["accuracy"]
        
        # Save model and scaler
        joblib.dump(fc_model, MODELS_DIR / "forecast_dl.pkl")
    else:
        print("Training HGB forecast model...")
        fc_model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        fc_model.fit(X_train_c, y_train_c)
        fc_pred = fc_model.predict(X_test_c)
        fc_f1 = f1_score(y_test_c, fc_pred, average="macro")
        fc_acc = accuracy_score(y_test_c, fc_pred)
        
        joblib.dump(fc_model, MODELS_DIR / "forecast_hgb.pkl")

    # === Explainability Model (RF for SHAP) ===
    rf_explain = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_explain.fit(X_train_c, y_train_c)

    joblib.dump(mon_model, MODELS_DIR / "monitoring_logreg.pkl")
    joblib.dump(rf_explain, MODELS_DIR / "explain_rf.pkl")

    joblib.dump(feature_cols_full, MODELS_DIR / "feature_cols_monitoring.pkl")
    joblib.dump(feature_cols_climate, MODELS_DIR / "feature_cols_forecast.pkl")

    df_full.to_csv(DATA_PROCESSED / "df_model_monitoring.csv", index=False)
    df_clim.to_csv(DATA_PROCESSED / "df_model_forecast.csv", index=False)

    metrics = {
        "monitoring_logreg": {"macro_f1": float(mon_f1), "accuracy": float(mon_acc)},
        "forecast_model": {
            "type": "HybridCNNLSTM" if use_dl else "HistGradientBoosting",
            "macro_f1": float(fc_f1),
            "accuracy": float(fc_acc),
        },
        "split_counts": {
            "monitoring": {"train": int(len(train_f)), "val": int(len(val_f)), "test": int(len(test_f))},
            "forecast": {"train": int(len(train_c)), "val": int(len(val_c)), "test": int(len(test_c))},
        },
        "notes": {
            "monitoring_model": "Uses risk_lag_* features; best for monitoring/alerts",
            "forecast_model": f"Climate + pruned humidity/NDVI production features. Type: {'DL (HybridCNNLSTM)' if use_dl else 'HGB'}",
            "dl_note": "DL architecture provides neural network backbone for deep learning project. Performance may vary from tree-based baseline.",
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    df = build_dataset()
    feature_cols_full, feature_cols_climate, df_full, df_clim = get_feature_sets(df)
    metrics = train_models(df_full, df_clim, feature_cols_full, feature_cols_climate, use_dl=True)
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Deep Learning Architecture (HybridCNNLSTM)")
    print("="*70)
    print("Saved models to:", MODELS_DIR)
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
