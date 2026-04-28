"""
Evaluate the **GRU** + Attention checkpoint on the held-out test period.
Mirrors preprocessing in `notebooks/deep_learning_model_selection.ipynb` and inference in `forecast_lstm.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED, FIG_DIR, MODELS_DIR, SEQUENCE_CHECKPOINT_NAME, TRAIN_END_YEAR, VAL_END_YEAR
from .lstm_risk_model import load_lstm_checkpoint
from .merge_dl_features import merge_auxiliary_features

LABEL_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Extreme"}


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(
            dict(year=out["year"].astype(int), month=out["month"].astype(int), day=1)
        )
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def make_sequences(
    df_part: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    city_col: str = "city",
    date_col: str = "date",
    target_col: str = "risk_label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_all, city_all, y_all = [], [], []
    for _, g in df_part.groupby(city_col):
        g = g.sort_values(date_col)
        if len(g) < seq_len:
            continue
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g[target_col].to_numpy(dtype=np.int64)
        cidx = g["city_idx"].to_numpy(dtype=np.int64)
        for t in range(seq_len - 1, len(g)):
            X_all.append(X[t - seq_len + 1 : t + 1])
            city_all.append(cidx[t])
            y_all.append(y[t])
    if not X_all:
        return np.zeros((0, seq_len, len(feature_cols))), np.array([]), np.array([])
    return np.stack(X_all), np.array(city_all), np.array(y_all)


def _gradient_input_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    city_idx: np.ndarray,
    feature_cols: list[str],
    device: torch.device,
    max_batches: int = 32,
    batch_size: int = 64,
) -> np.ndarray:
    """Mean |x * grad| w.r.t. x, averaged over batch and time (simple input-gradient saliency)."""
    model.eval()
    n = min(len(X), max_batches * batch_size)
    acc = np.zeros(len(feature_cols), dtype=np.float64)
    count = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device, requires_grad=True)
        cb = torch.tensor(city_idx[start:end], dtype=torch.long, device=device)
        logits = model(xb, cb)
        target = logits.argmax(dim=1)
        sel = logits.gather(1, target.unsqueeze(1)).sum()
        model.zero_grad(set_to_none=True)
        sel.backward()
        g = xb.grad.detach()
        sal = (g.abs() * xb.detach().abs()).mean(dim=(0, 1)).cpu().numpy()
        acc += sal * (end - start)
        count += end - start
    return acc / max(count, 1)


def _predict_batches(
    model: torch.nn.Module,
    X_test: np.ndarray,
    c_test: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.tensor(X_test[i : i + batch_size], dtype=torch.float32, device=device)
            cb = torch.tensor(c_test[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb, cb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_lstm(checkpoint_name: str | None = None) -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ck_name = checkpoint_name or SEQUENCE_CHECKPOINT_NAME
    ck = MODELS_DIR / ck_name
    if not ck.is_file():
        raise FileNotFoundError(f"Missing GRU checkpoint: {ck}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_lstm_checkpoint(ck, device=device)
    feature_cols: list[str] = list(payload["feature_cols"])
    city_to_idx: dict = payload["city_to_idx"]
    cfg: dict = payload.get("config", {})
    seq_len = int(cfg.get("seq_len", 12))

    df_base = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    df_m = merge_auxiliary_features(df_base)
    df_m = _ensure_date(df_m)
    df_m["city"] = df_m["city"].astype(str).str.strip()
    df_m["year_tmp"] = df_m["year"].astype(int)

    for c in feature_cols:
        if c not in df_m.columns:
            df_m[c] = np.nan

    df_m["risk_label"] = pd.to_numeric(df_m["risk_label"], errors="coerce")
    df_m = df_m.dropna(subset=["risk_label"]).copy()
    df_m["risk_label"] = df_m["risk_label"].astype(int)

    unknown = set(df_m["city"].unique()) - set(city_to_idx.keys())
    if unknown:
        df_m = df_m[df_m["city"].isin(city_to_idx.keys())].copy()

    df_m["city_idx"] = df_m["city"].map(city_to_idx)
    if df_m["city_idx"].isna().any():
        df_m = df_m.dropna(subset=["city_idx"]).copy()
    df_m["city_idx"] = df_m["city_idx"].astype(int)

    train_mask = df_m["year_tmp"] <= TRAIN_END_YEAR
    df_m[feature_cols] = df_m[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_m[feature_cols] = df_m[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = df_m.loc[train_mask, feature_cols].median(numeric_only=True)
    df_m[feature_cols] = df_m[feature_cols].fillna(medians)
    df_m[feature_cols] = df_m[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(df_m.loc[train_mask, feature_cols].to_numpy(dtype=np.float64))
    scaled = scaler.transform(df_m[feature_cols].to_numpy(dtype=np.float64))
    df_m[feature_cols] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    df_m = df_m.sort_values(["city", "date"]).reset_index(drop=True)

    test_df = df_m[df_m["year_tmp"] > VAL_END_YEAR].copy()
    trainval_df = df_m[df_m["year_tmp"] <= VAL_END_YEAR].copy()
    y_trainval = trainval_df["risk_label"].to_numpy()

    baseline_class = int(pd.Series(y_trainval).value_counts().idxmax())
    X_test, c_test, y_test = make_sequences(
        test_df, feature_cols, seq_len, city_col="city", date_col="date", target_col="risk_label"
    )
    if len(y_test) == 0:
        raise RuntimeError("No test sequences; check df_model_forecast.csv and year splits.")

    baseline_pred = np.full_like(y_test, fill_value=baseline_class)

    pred = _predict_batches(model, X_test, c_test, device)

    def metrics_row(name: str, y_true, y_p) -> dict:
        return {
            "model": name,
            "accuracy": float(accuracy_score(y_true, y_p)),
            "macro_f1": float(f1_score(y_true, y_p, average="macro")),
            "macro_precision": float(precision_score(y_true, y_p, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_p, average="macro", zero_division=0)),
            "extreme_recall": float(
                recall_score((y_true == 3).astype(int), (y_p == 3).astype(int), zero_division=0)
            ),
        }

    rows = [
        metrics_row("baseline_majority", y_test, baseline_pred),
        metrics_row(str(payload.get("model_name", "GRU_Attn")), y_test, pred),
    ]

    metrics = pd.DataFrame(rows)

    report_b = classification_report(
        y_test, baseline_pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0
    )
    (FIG_DIR / "classification_report_baseline.txt").write_text(report_b)
    ConfusionMatrixDisplay.from_predictions(y_test, baseline_pred, normalize="true")
    plt.title("Baseline (majority class) – test set")
    plt.savefig(FIG_DIR / "confusion_matrix_baseline.png", dpi=220, bbox_inches="tight")
    plt.close()

    report = classification_report(
        y_test, pred, target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0
    )
    mname = str(payload.get("model_name", "GRU_Attn"))
    (FIG_DIR / "classification_report_sequence.txt").write_text(report)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, normalize="true")
    plt.title(f"{mname} – normalized confusion matrix (test)")
    plt.savefig(FIG_DIR / "confusion_matrix_sequence.png", dpi=220, bbox_inches="tight")
    plt.close()

    metrics.to_csv(FIG_DIR / "model_metrics.csv", index=False)
    best = metrics.sort_values(["macro_f1", "extreme_recall", "accuracy"], ascending=False).iloc[0]
    (FIG_DIR / "best_model.txt").write_text(best.to_string())
    print(metrics.to_string(index=False))
    print("\nBEST (selection rule: macro_f1, extreme_recall, accuracy):\n", best.to_string())

    try:
        imp = _gradient_input_importance(model, X_test, c_test, feature_cols, device)
        order = np.argsort(-imp)[:15]
        names = [feature_cols[i] for i in order]
        vals = imp[order]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names[::-1], vals[::-1], color="steelblue")
        ax.set_xlabel("Mean |input × ∂logit| (test subsample)")
        ax.set_title(f"{mname}: input-gradient feature saliency (top 15)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "sequence_feature_saliency_top15.png", dpi=220, bbox_inches="tight")
        plt.close()
        pd.DataFrame({"feature": feature_cols, "saliency": imp}).sort_values("saliency", ascending=False).to_csv(
            FIG_DIR / "sequence_feature_saliency.csv", index=False
        )
    except Exception as e:
        print("Skipping saliency plot:", e)

    return metrics


def main():
    evaluate_lstm()


if __name__ == "__main__":
    main()
