import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from .config import MODELS_DIR, FIG_DIR, DATA_PROCESSED


def save_bar(series: pd.Series, out_png, title, topn=15):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = series.sort_values(ascending=False).head(topn)
    ax = s[::-1].plot(kind="barh")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load forecast dataset + features
    df = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    feat_fc = joblib.load(MODELS_DIR / "feature_cols_forecast.pkl")

    # Split to use TEST only (what you report)
    test = df[df["year"] > 2019].copy()
    X_test = test[feat_fc]
    y_test = test["risk_label"]

    # Load models
    fc_model = joblib.load(MODELS_DIR / "forecast_hgb.pkl")     # best deploy model
    rf_exp   = joblib.load(MODELS_DIR / "explain_rf.pkl")       # explainability model

    # --------------------------
    # A) Permutation Importance on BEST model (HGB)  ✅
    # --------------------------
    X_s = X_test.sample(min(600, len(X_test)), random_state=42)
    y_s = y_test.loc[X_s.index]

    perm = permutation_importance(
        fc_model, X_s, y_s,
        n_repeats=15,
        random_state=42,
        n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=feat_fc).sort_values(ascending=False)
    perm_imp.to_csv(FIG_DIR / "perm_importance_forecast_hgb.csv")

    save_bar(
        perm_imp,
        FIG_DIR / "perm_importance_forecast_hgb_top15.png",
        "Permutation Importance (Forecast HGB) – Top 15 Features",
        topn=15
    )

    # --------------------------
    # B) Built-in RF feature importance (quick global story) ✅
    # --------------------------
    rf_imp = pd.Series(rf_exp.feature_importances_, index=feat_fc).sort_values(ascending=False)
    rf_imp.to_csv(FIG_DIR / "rf_feature_importance.csv")

    save_bar(
        rf_imp,
        FIG_DIR / "rf_feature_importance_top15.png",
        "Random Forest Feature Importance – Top 15 Features",
        topn=15
    )

    print("Saved feature importance outputs to:", FIG_DIR)
    print("Top 10 (Permutation, Forecast HGB):")
    print(perm_imp.head(10))


if __name__ == "__main__":
    main()
