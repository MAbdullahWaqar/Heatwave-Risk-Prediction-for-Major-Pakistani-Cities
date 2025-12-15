import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, FIG_DIR, DATA_PROCESSED


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    feat = joblib.load(MODELS_DIR / "feature_cols_forecast.pkl")
    rf = joblib.load(MODELS_DIR / "explain_rf.pkl")

    # Use last years as test-ish subset
    X = df[df["year"] > 2019][feat].copy()
    X = X.sample(min(300, len(X)), random_state=42)

    import shap
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X)

    # Handle common multiclass output shapes
    # sv may be list[class] or array[samples, features, classes]
    if isinstance(sv, list):
        sv_ext = np.array(sv[3])
    else:
        arr = np.array(sv)
        sv_ext = arr[:, :, 3]  # (samples, features) for class 3 Extreme

    # Save summary plot
    shap.summary_plot(sv_ext, X, show=False)
    import matplotlib.pyplot as plt
    plt.title("SHAP Summary (Extreme Risk Class)")
    plt.savefig(FIG_DIR / "shap_summary_extreme.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved SHAP summary to:", FIG_DIR / "shap_summary_extreme.png")
    
        # Save one waterfall example (first sample)
    exp = explainer(X.iloc[[0]])
    vals = exp.values
    base = exp.base_values

    if vals.ndim == 3:  # (1, features, classes)
        vals_c = vals[0, :, 3]
        base_c = base[0, 3]
    else:
        vals_c = vals[0]
        base_c = base[0]

    e = shap.Explanation(values=vals_c, base_values=base_c, data=X.iloc[0], feature_names=feat)
    shap.plots.waterfall(e, show=False)
    import matplotlib.pyplot as plt
    plt.title("Local SHAP Waterfall (Extreme Risk)")
    plt.savefig(FIG_DIR / "shap_waterfall_example.png", dpi=220, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()
