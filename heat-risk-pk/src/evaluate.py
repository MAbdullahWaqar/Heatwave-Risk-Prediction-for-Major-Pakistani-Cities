import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)

from .config import MODELS_DIR, FIG_DIR, DATA_PROCESSED
from .model_zoo import get_models


LABEL_NAMES = {0:"Low", 1:"Moderate", 2:"High", 3:"Extreme"}

def save_report(text, path):
    path.write_text(text)

def evaluate_one(name, model, X_train, y_train, X_test, y_test, out_prefix):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Core metrics (macro handles imbalance)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    macro_prec = precision_score(y_test, pred, average="macro", zero_division=0)
    macro_rec  = recall_score(y_test, pred, average="macro", zero_division=0)

    # Extreme class recall (important stakeholder metric)
    ext_rec = recall_score((y_test==3).astype(int), (pred==3).astype(int), zero_division=0)

    # Save classification report
    report = classification_report(y_test, pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0)
    save_report(report, out_prefix.with_suffix(".txt"))

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, pred, normalize="true")
    plt.title(f"{name} – Normalized Confusion Matrix")
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close()

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "extreme_recall": ext_rec
    }

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Choose which dataset to evaluate on:
    # Forecast dataset = climate-only (recommended for "best prediction model")
    df = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")

    feature_cols = joblib.load(MODELS_DIR / "feature_cols_forecast.pkl")

    # Split (match your earlier setup)
    train = df[df["year"] <= 2015].copy()
    val   = df[(df["year"] > 2015) & (df["year"] <= 2019)].copy()
    test  = df[df["year"] > 2019].copy()

    # Train on train+val, test on test (final evaluation)
    trainval = pd.concat([train, val], ignore_index=True)

    X_train = trainval[feature_cols]
    y_train = trainval["risk_label"]
    X_test  = test[feature_cols]
    y_test  = test["risk_label"]

    # Baseline (persistence) not valid here because forecast set has NO risk_lags.
    # We'll create a naive baseline: predict "same month climatology class"
    # simplest baseline: always predict the most common class in trainval
    baseline_class = y_train.value_counts().idxmax()
    baseline_pred = np.full_like(y_test, fill_value=baseline_class)

    # Save baseline report/CM
    b_report = classification_report(y_test, baseline_pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0)
    save_report(b_report, FIG_DIR / "classification_report_baseline.txt")
    ConfusionMatrixDisplay.from_predictions(y_test, baseline_pred, normalize="true")
    plt.title("Baseline (Most Frequent Class) – Normalized Confusion Matrix")
    plt.savefig(FIG_DIR / "confusion_matrix_baseline.png", dpi=220, bbox_inches="tight")
    plt.close()

    rows = [{
        "model": "baseline_majority",
        "accuracy": accuracy_score(y_test, baseline_pred),
        "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
        "macro_precision": precision_score(y_test, baseline_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, baseline_pred, average="macro", zero_division=0),
        "extreme_recall": recall_score((y_test==3).astype(int), (baseline_pred==3).astype(int), zero_division=0),
    }]

    models = get_models(kind="forecast")

    for k, m in models.items():
        out_prefix = FIG_DIR / f"confusion_matrix_{k}"
        rows.append(evaluate_one(k, m, X_train, y_train, X_test, y_test, out_prefix))
        print("Done:", k)

    metrics = pd.DataFrame(rows).sort_values(["macro_f1","extreme_recall"], ascending=False)
    metrics.to_csv(FIG_DIR / "model_metrics.csv", index=False)
    print(metrics)

    # Best model selection rule (you can justify this in report):
    # Primary: Macro-F1, tie-breaker: Extreme Recall, then Accuracy
    best = metrics.sort_values(["macro_f1","extreme_recall","accuracy"], ascending=False).iloc[0]
    (FIG_DIR / "best_model.txt").write_text(best.to_string())
    print("\nBEST MODEL:\n", best)

if __name__ == "__main__":
    main()
