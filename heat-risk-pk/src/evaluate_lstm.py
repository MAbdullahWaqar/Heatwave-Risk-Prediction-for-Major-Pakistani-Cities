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


def _month_to_period(m: int) -> str:
    if m in (11, 12, 1, 2):
        return "Winter (Nov-Feb)"
    if m in (3, 4):
        return "Spring (Mar-Apr)"
    if m in (5, 6, 7):
        return "Peak Summer (May-Jul)"
    if m in (8, 9):
        return "Monsoon (Aug-Sep)"
    return "Autumn (Oct)"


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
    idx_to_city = {int(v): str(k) for k, v in city_to_idx.items()}

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

    # City-wise evaluation summary for frontend table.
    def _city_notes(acc: float, extreme_rec: float) -> tuple[str, str]:
        if acc >= 0.93 and extreme_rec >= 0.85:
            return (
                "Stable seasonal pattern; model confidence high.",
                "Low risk: keep routine monitoring.",
            )
        if extreme_rec < 0.20:
            return (
                "Rare/extreme events under-detected in this city.",
                "Lower extreme threshold or add city-specific tuning.",
            )
        if acc < 0.80:
            return (
                "High seasonal variability and class overlap.",
                "Add city-specific features and monitor transition months.",
            )
        if acc < 0.90:
            return (
                "Moderate misclassification near class boundaries.",
                "Focus monitoring around seasonal transition periods.",
            )
        return (
            "Good overall fit with occasional boundary errors.",
            "Maintain current model; review edge cases quarterly.",
        )

    city_rows = []
    c_test_s = pd.Series(c_test).astype(int)
    y_true_s = pd.Series(y_test).astype(int)
    y_pred_s = pd.Series(pred).astype(int)
    for cidx in sorted(c_test_s.unique().tolist()):
        m = c_test_s == cidx
        yt = y_true_s[m].to_numpy()
        yp = y_pred_s[m].to_numpy()
        acc = float(accuracy_score(yt, yp))
        macro = float(f1_score(yt, yp, average="macro", zero_division=0))
        ext_rec = float(
            recall_score((yt == 3).astype(int), (yp == 3).astype(int), zero_division=0)
        )
        challenge, recommendation = _city_notes(acc, ext_rec)
        city_rows.append(
            {
                "city": idx_to_city.get(int(cidx), f"city_{cidx}"),
                "samples": int(m.sum()),
                "accuracy": acc,
                "macro_f1": macro,
                "extreme_recall": ext_rec,
                "challenge": challenge,
                "recommendation": recommendation,
            }
        )
    city_df = pd.DataFrame(city_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    city_df.to_csv(FIG_DIR / "city_wise_accuracy.csv", index=False)

    # Most common misclassifications for frontend error analysis cards.
    cm = pd.crosstab(
        pd.Series(y_test, name="true"),
        pd.Series(pred, name="pred"),
        dropna=False,
    ).reindex(index=range(4), columns=range(4), fill_value=0)
    total_err = int((pred != y_test).sum())
    pair_rows = []
    if total_err > 0:
        for t in range(4):
            for p_ in range(4):
                if t == p_:
                    continue
                cnt = int(cm.loc[t, p_])
                if cnt <= 0:
                    continue
                pair_rows.append(
                    {
                        "from_class": LABEL_NAMES[t],
                        "to_class": LABEL_NAMES[p_],
                        "count": cnt,
                        "pct_of_errors": round(100.0 * cnt / total_err, 1),
                    }
                )
    pair_df = pd.DataFrame(pair_rows).sort_values("count", ascending=False).head(3).reset_index(drop=True)
    if len(pair_df) > 0:
        pair_df.to_csv(FIG_DIR / "most_common_misclassifications.csv", index=False)
    else:
        pd.DataFrame(
            columns=["from_class", "to_class", "count", "pct_of_errors"]
        ).to_csv(FIG_DIR / "most_common_misclassifications.csv", index=False)

    # Binary error summary for Extreme class (used by frontend temporal cards).
    is_ext_true = (y_test == 3).astype(int)
    is_ext_pred = (pred == 3).astype(int)
    tp = int(((is_ext_true == 1) & (is_ext_pred == 1)).sum())
    fn = int(((is_ext_true == 1) & (is_ext_pred == 0)).sum())
    fp = int(((is_ext_true == 0) & (is_ext_pred == 1)).sum())
    tn = int(((is_ext_true == 0) & (is_ext_pred == 0)).sum())
    total_ext = int(is_ext_true.sum())
    total_non_ext = int((is_ext_true == 0).sum())
    miss_rate = float(fn / max(total_ext, 1))
    false_alarm_rate = float(fp / max((tp + fp), 1))
    precision_ext = float(tp / max((tp + fp), 1))
    recall_ext = float(tp / max((tp + fn), 1))
    binary_df = pd.DataFrame(
        [
            {
                "total_samples": int(len(y_test)),
                "total_extreme_true": total_ext,
                "tp_extreme": tp,
                "fn_extreme": fn,
                "fp_extreme": fp,
                "tn_non_extreme": tn,
                "miss_rate_extreme": miss_rate,
                "false_alarm_rate_extreme": false_alarm_rate,
                "precision_extreme": precision_ext,
                "recall_extreme": recall_ext,
                "val_accuracy": float(np.nan),  # optional placeholder for later extensions
            }
        ]
    )
    binary_df.to_csv(FIG_DIR / "extreme_error_summary.csv", index=False)

    # Class-specific performance summary for frontend cards.
    class_rows = []
    for cls in range(4):
        yt_bin = (y_test == cls).astype(int)
        yp_bin = (pred == cls).astype(int)
        support = int(yt_bin.sum())
        acc_cls = float((yt_bin == yp_bin).mean())
        rec_cls = float(recall_score(yt_bin, yp_bin, zero_division=0))
        prec_cls = float(precision_score(yt_bin, yp_bin, zero_division=0))
        err_rate = 1.0 - rec_cls
        if cls == 0:
            notes = "Lowest-risk class; often confused around seasonal transitions."
        elif cls == 1:
            notes = "Intermediate class with widest overlap to Low/High."
        elif cls == 2:
            notes = "Boundary overlap with Moderate and Extreme months."
        else:
            notes = "Critical class; extreme-event detection quality."
        class_rows.append(
            {
                "class_id": cls,
                "class_name": LABEL_NAMES[cls],
                "data_share_pct": round(100.0 * support / len(y_test), 1),
                "support": support,
                "accuracy": acc_cls,
                "recall": rec_cls,
                "precision": prec_cls,
                "error_rate": err_rate,
                "notes": notes,
            }
        )
    pd.DataFrame(class_rows).to_csv(FIG_DIR / "class_specific_performance.csv", index=False)

    # Temporal error patterns by season-like periods.
    test_seq = test_df.copy()
    test_seq["_month_int"] = test_seq["month"].astype(int)
    # Sequence label corresponds to end-of-window row.
    seq_meta = []
    for _, g in test_seq.groupby("city"):
        g = g.sort_values("date")
        if len(g) < seq_len:
            continue
        for t in range(seq_len - 1, len(g)):
            seq_meta.append({"month": int(g.iloc[t]["_month_int"])})
    meta_df = pd.DataFrame(seq_meta)
    if len(meta_df) == len(y_test):
        meta_df["y_true"] = y_test
        meta_df["y_pred"] = pred
        meta_df["period"] = meta_df["month"].map(_month_to_period)
        trows = []
        for p_name, g in meta_df.groupby("period"):
            acc_p = float((g["y_true"] == g["y_pred"]).mean())
            miss_ext = float(
                ((g["y_true"] == 3) & (g["y_pred"] != 3)).sum() / max((g["y_true"] == 3).sum(), 1)
            )
            if acc_p >= 0.9:
                note = "Consistent patterns; relatively easier predictions."
            elif miss_ext > 0.4:
                note = "Higher extreme miss rate; monitor warning thresholds."
            else:
                note = "Moderate overlap between adjacent classes."
            trows.append(
                {
                    "period": p_name,
                    "samples": int(len(g)),
                    "accuracy": acc_p,
                    "extreme_miss_rate": miss_ext,
                    "notes": note,
                }
            )
        pd.DataFrame(trows).sort_values("accuracy", ascending=False).to_csv(
            FIG_DIR / "temporal_error_patterns.csv", index=False
        )

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
