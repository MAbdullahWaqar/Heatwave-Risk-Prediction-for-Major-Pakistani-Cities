"""
Generate GRU SHAP artifacts used by the Streamlit frontend.

Outputs written to outputs/figures:
- shap_summary_extreme.png
- shap_waterfall_example.png
- shap_feature_importance_best_model.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED, FIG_DIR, MODELS_DIR, SEQUENCE_CHECKPOINT_NAME, TRAIN_END_YEAR, VAL_END_YEAR
from .lstm_risk_model import load_lstm_checkpoint
from .merge_dl_features import merge_auxiliary_features
from .evaluate_lstm import _ensure_date, make_sequences


def build_eval_table(feature_cols: list[str], city_to_idx: dict[str, int]) -> pd.DataFrame:
    df_base = pd.read_csv(DATA_PROCESSED / "df_model_forecast.csv")
    df_m = merge_auxiliary_features(df_base)
    df_m = _ensure_date(df_m)
    df_m["city"] = df_m["city"].astype(str).str.strip()
    df_m = df_m[df_m["city"].isin(city_to_idx.keys())].copy()
    df_m["city_idx"] = df_m["city"].map(city_to_idx).astype(int)
    df_m["year_tmp"] = df_m["year"].astype(int)

    for c in feature_cols:
        if c not in df_m.columns:
            df_m[c] = np.nan
    df_m[feature_cols] = df_m[feature_cols].apply(pd.to_numeric, errors="coerce")
    df_m[feature_cols] = df_m[feature_cols].replace([np.inf, -np.inf], np.nan)

    train_mask = df_m["year_tmp"] <= TRAIN_END_YEAR
    medians = df_m.loc[train_mask, feature_cols].median(numeric_only=True)
    df_m[feature_cols] = df_m[feature_cols].fillna(medians).fillna(0.0)

    scaler = StandardScaler()
    scaler.fit(df_m.loc[train_mask, feature_cols].to_numpy(dtype=np.float64))
    df_m[feature_cols] = scaler.transform(df_m[feature_cols].to_numpy(dtype=np.float64))
    return df_m.sort_values(["city", "date"]).reset_index(drop=True)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ck = MODELS_DIR / SEQUENCE_CHECKPOINT_NAME
    if not ck.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ck}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_lstm_checkpoint(ck, device=device)
    model.eval()

    feature_cols: list[str] = list(payload["feature_cols"])
    city_to_idx: dict[str, int] = payload["city_to_idx"]
    seq_len = int(payload.get("config", {}).get("seq_len", 12))

    df_m = build_eval_table(feature_cols, city_to_idx)
    test_df = df_m[df_m["year_tmp"] > VAL_END_YEAR].copy()
    X_test, c_test, _ = make_sequences(test_df, feature_cols, seq_len)
    if len(X_test) == 0:
        raise RuntimeError("No test sequences available for SHAP generation.")

    # Small sample sizes keep SHAP generation practical.
    rng = np.random.default_rng(42)
    # Use one representative city for stable SHAP shape/output.
    city_values, city_counts = np.unique(c_test, return_counts=True)
    city_id = int(city_values[np.argmax(city_counts)])
    city_mask = c_test == city_id
    X_city = X_test[city_mask]
    if len(X_city) < 10:
        X_city = X_test

    bg_n = min(80, len(X_city))
    ex_n = min(120, len(X_city))
    bg_idx = rng.choice(len(X_city), size=bg_n, replace=False)
    ex_idx = rng.choice(len(X_city), size=ex_n, replace=False)

    xb = torch.tensor(X_city[bg_idx], dtype=torch.float32, device=device)
    xe = torch.tensor(X_city[ex_idx], dtype=torch.float32, device=device)

    class ExtremeProbWrapper(nn.Module):
        def __init__(self, base_model: nn.Module, city_id: int):
            super().__init__()
            self.base_model = base_model
            self.city_id = int(city_id)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cb = torch.full((x.shape[0],), self.city_id, dtype=torch.long, device=x.device)
            logits = self.base_model(x, cb)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 3:4]  # extreme probability

    wrapped = ExtremeProbWrapper(model, city_id).to(device)
    explainer = shap.GradientExplainer(wrapped, xb)
    sv = explainer.shap_values(xe)
    arr = np.asarray(sv)
    if arr.ndim == 4:
        # Common shapes:
        # - [batch, seq_len, n_feat, 1]
        # - [1, batch, seq_len, n_feat]
        if arr.shape[-1] == 1:
            contrib_batch = arr[..., 0]
        else:
            contrib_batch = arr[0]
    elif arr.ndim == 3:
        # [batch, seq_len, n_feat]
        contrib_batch = arr
    elif arr.ndim == 2:
        # [batch, n_feat]
        contrib_batch = arr[:, None, :]
    else:
        raise RuntimeError(f"Unexpected SHAP shape: {arr.shape}")

    signed_mat = contrib_batch.mean(axis=1)  # [batch, n_feat]
    abs_mat = np.abs(signed_mat)
    xmean_mat = xe.detach().cpu().numpy().mean(axis=1)

    # Waterfall for first explained sample.
    ex = shap.Explanation(
        values=signed_mat[0],
        base_values=0.0,
        data=xmean_mat[0],
        feature_names=feature_cols,
    )
    shap.plots.waterfall(ex, show=False, max_display=15)
    plt.title("SHAP Waterfall (Extreme Risk Example)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_example.png", dpi=220, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(abs_mat, axis=0)
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(FIG_DIR / "shap_feature_importance_best_model.csv", index=False)

    # Summary beeswarm-style plot from tabularized feature means for readability.
    # Use top 15 features to keep chart compact.
    top_features = imp_df.head(15)["feature"].tolist()
    top_idx = [feature_cols.index(f) for f in top_features]
    X_top = xmean_mat[:, top_idx]
    SV_top = signed_mat[:, top_idx]
    shap.summary_plot(
        SV_top,
        X_top,
        feature_names=top_features,
        show=False,
        max_display=15,
    )
    plt.title("SHAP Summary (Extreme Risk Class)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_extreme.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved:", FIG_DIR / "shap_summary_extreme.png")
    print("Saved:", FIG_DIR / "shap_waterfall_example.png")
    print("Saved:", FIG_DIR / "shap_feature_importance_best_model.csv")


if __name__ == "__main__":
    main()

