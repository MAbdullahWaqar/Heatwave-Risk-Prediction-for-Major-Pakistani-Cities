import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from .config import MODELS_DIR, FIG_DIR, DATA_PROCESSED
from .model_zoo import get_models


LABEL_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}


def save_report(text, path):
    path.write_text(text)


def evaluate_one(name, model, X_train, y_train, X_test, y_test, out_prefix):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    macro_prec = precision_score(y_test, pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, pred, average="macro", zero_division=0)
    ext_rec = recall_score((y_test == 3).astype(int), (pred == 3).astype(int), zero_division=0)

    report = classification_report(
        y_test, pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0
    )
    save_report(report, out_prefix.with_suffix(".txt"))

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
        "extreme_recall": ext_rec,
    }


def main_tabular_baselines():
    """Optional: sklearn models on flattened features (does not overwrite LSTM primary metrics)."""
    out_dir = FIG_DIR / "tabular_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols_forecast.pkl")

    train = df[df["year"] <= 2015].copy()
    val = df[(df["year"] > 2015) & (df["year"] <= 2019)].copy()
    test = df[df["year"] > 2019].copy()
    trainval = pd.concat([train, val], ignore_index=True)

    X_train = trainval[feature_cols]
    y_train = trainval["risk_label"]
    X_test = test[feature_cols]
    y_test = test["risk_label"]

    baseline_class = y_train.value_counts().idxmax()
    baseline_pred = np.full_like(y_test, fill_value=baseline_class)

    b_report = classification_report(
        y_test, baseline_pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0
    )
    save_report(b_report, out_dir / "classification_report_baseline.txt")
    ConfusionMatrixDisplay.from_predictions(y_test, baseline_pred, normalize="true")
    plt.title("Baseline (Most Frequent Class) – Normalized Confusion Matrix")
    plt.savefig(out_dir / "confusion_matrix_baseline.png", dpi=220, bbox_inches="tight")
    plt.close()

    rows = [
        {
            "model": "baseline_majority",
            "accuracy": accuracy_score(y_test, baseline_pred),
            "macro_f1": f1_score(y_test, baseline_pred, average="macro"),
            "macro_precision": precision_score(y_test, baseline_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, baseline_pred, average="macro", zero_division=0),
            "extreme_recall": recall_score(
                (y_test == 3).astype(int), (baseline_pred == 3).astype(int), zero_division=0
            ),
        }
    ]

    models = get_models(kind="forecast")
    for k, m in models.items():
        out_prefix = out_dir / f"confusion_matrix_{k}"
        rows.append(evaluate_one(k, m, X_train, y_train, X_test, y_test, out_prefix))
        print("Done:", k)

    metrics = pd.DataFrame(rows).sort_values(["macro_f1", "extreme_recall"], ascending=False)
    metrics.to_csv(out_dir / "model_metrics_tabular.csv", index=False)
    print(metrics)
    print("Saved tabular baseline figures to:", out_dir)


def main():
    """Primary evaluation: GRU + Attention (same data contract as forecasts)."""
    from .evaluate_lstm import evaluate_lstm

    evaluate_lstm()


if __name__ == "__main__":
    if "--tabular" in sys.argv:
        main_tabular_baselines()
    else:
        main()
