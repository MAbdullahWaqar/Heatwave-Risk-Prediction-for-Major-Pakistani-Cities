# Deep learning model guide (GRU + attention)

This document explains **how the heat-risk model works** end to end: data going in, the network, scores, accuracy, how the best model is chosen, and how that compares to the **legacy tabular** forecaster. It matches:

- **`heat-risk-pk/notebooks/deep_learning_model_selection.ipynb`** — training, metrics, and model comparison (GRU, LSTM, TCN, Transformer)  
- **`heat-risk-pk/models/gru_attn_best.pkl`** — **GRU** checkpoint (`torch.save` / `torch.load` dict with `model_state_dict`, `feature_cols`, `city_to_idx`, `config`, `model_name`). Override with env **`SEQUENCE_CHECKPOINT_NAME`** only for experiments (see `src/config.py`).

For install, folder layout, and full pipeline commands, see the main **[README.md](README.md)**.

---

## Contents

1. [Input pipeline](#1-input-pipeline)  
2. [How the model works](#2-how-the-model-works)  
3. [Model output](#3-model-output)  
4. [Score and loss calculation](#4-score-and-loss-calculation)  
5. [What “accuracy” means here](#5-what-accuracy-means-here)  
6. [How the best model is chosen (and what counts as a baseline)](#6-how-the-best-model-is-chosen-and-what-counts-as-a-baseline)  
7. [Comparison with the legacy (tabular) edition](#7-comparison-with-the-legacy-tabular-edition)

---

## 1. Input pipeline

Everything below follows the notebook’s **`Config`** and loading cells.

### 1.1 Files the notebook reads

| Config field | Path (from `heat-risk-pk/notebooks/`) | Role |
|--------------|----------------------------------------|------|
| `data_path` | `../data/processed/df_model_forecast.csv` | **Required.** City–month table with `risk_label`, `city`, date or `year`/`month`, and numeric features. |
| `humidity_path` | `../data/raw/pakistan_humidity_daily.csv` | **Optional** (`os.path.exists`). Daily humidity aggregated to monthly `hum_*` columns. |
| `ndvi_path` | `../data/raw/pakistan_ndvi_monthly.csv` | **Optional.** NDVI → `ndvi_monthly`. |
| `merged_weather_scaled_path` | `../data/processed/pakistan_weather_merged_scaled.csv` | **Optional.** Merged weather → `wm_*` columns. |

The notebook **does not** open raw weather CSVs or World Bank files directly; those feeds are already reflected **inside** `df_model_forecast.csv` when that file is built (e.g. via `src/train.py`).

### 1.2 Feature selection

After merges, the notebook keeps **numeric** columns as candidate features and **excludes** metadata and leakage: `risk_label`, `city`, `date`, `year`, `month`, and **`risk_lag_*`**. The remaining list is **`feature_cols`** (saved in the checkpoint).

### 1.3 Train / validation / test split

Uses a `year_tmp` column from the date:

- **Train:** `year_tmp ≤ train_end_year` (default **2015**)  
- **Validation:** `train_end_year < year_tmp ≤ val_end_year` (**2016–2019**)  
- **Test:** `year_tmp > val_end_year` (**≥ 2020**)

### 1.4 Imputation and scaling

- **Median imputation** per feature using **training rows only**; any remaining NaNs are filled with **0**.  
- **`StandardScaler`**: **fit on training rows only**, then applied to the full dataframe (and the same logic is mirrored in `evaluate_lstm.py` / `forecast_lstm.py` for parity).

### 1.5 Building sequences

For **each city**, rows are sorted by time. For each end index `t` (from `seq_len - 1` to the last row):

- **`X`**: the last **`seq_len`** rows (default **12 months**) of **`feature_cols`** → shape `(seq_len, n_features)`  
- **`y`**: **`risk_label`** at month `t` (the **end** of the window)  
- **`city_idx`**: integer id for that city from **`city_to_idx`** (saved in the checkpoint)

The notebook’s `build_loaders` wraps these in PyTorch **`DataLoader`** batches: **`(xb, cb, yb)`** = sequences, city indices, labels.

---

## 2. How the model works

The model class is **`RNNAttentionClassifier`** in **`heat-risk-pk/src/lstm_risk_model.py`**. **Production** uses **`torch.nn.GRU`** (payload **`GRU_Attn`** in **`gru_attn_best.pkl`**). The notebook also trains **`LSTM_Attn`** and other candidates for the same competition metric; only the **GRU** checkpoint is wired into **`forecast_lstm.py`** and the app by default.

### 2.1 Forward pass (step by step)

1. **`xb`**: shape **`(batch, seq_len, n_features)`** — scaled feature vectors for each month in the window.  
2. **`cb`**: shape **`(batch,)`** — city index per sample → **`nn.Embedding(num_cities, embed_dim)`** (default `embed_dim = 8`).  
3. **`nn.GRU`** with **`bidirectional=True`**: two passes over the **same fixed window** (forward and backward in time within those 12 months). Outputs are **concatenated** per step → hidden size **`2 × hidden_dim`**.  
4. **`AttentionPool`**: a small linear layer scores each time step → **softmax over time** → **weighted sum** of GRU outputs → one **context** vector per sample.  
5. **Classifier head**: **[context ∥ city_embedding]** → `Linear → ReLU → Dropout → Linear` → **`num_classes` logits** (four risk classes: Low, Moderate, High, Extreme).

### 2.2 Bidirectional GRU (no future labels)

**Bidirectional** means two chains over the **context window** so each step can use both past and future **months inside the window**. The **label** is always for the **last** month of the window; no future **labels** are used.

---

## 3. Model output

- **Raw output:** **`logits`** of shape `(batch, 4)` — unnormalized scores for the four classes.  
- **Probabilities:** **`softmax(logits, dim=1)`** — four nonnegative values summing to 1 (used in forecasting CSVs).  
- **Predicted class (hard label):** **`argmax(logits, dim=1)`** — integer **0–3** (same rule as in the notebook’s `run_epoch`).

### What is stored in the GRU checkpoint (`gru_attn_best.pkl`)

Typical keys:

- **`model_state_dict`** — all learnable weights  
- **`feature_cols`** — ordered feature names (must match inference rows)  
- **`city_to_idx`** — string city name → integer id  
- **`config`** — hyperparameters (`seq_len`, `hidden_dim`, `lr`, etc.)  
- **`model_name`** — e.g. **`GRU_Attn`**

Probabilities are **not** stored in the file; they are **computed at runtime** from the loaded weights and input tensor.

---

## 4. Score and loss calculation

### 4.1 Training loss

- **`nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`**  
- **`class_weights`**: from **training** label counts — inverse-frequency style, normalized so the mean weight is 1 (rarer classes get higher weight).  
- **`label_smoothing=0.05`**: regularizes the target distribution slightly.

### 4.2 Metrics computed in `run_epoch`

After looping all batches, the notebook concatenates **`all_true`** and **`all_preds`** (predictions = **`argmax(logits)`**) and computes:

| Metric | Definition (sklearn) | Meaning |
|--------|----------------------|--------|
| **Macro F1** | `f1_score(..., average="macro", zero_division=0)` | Average of **per-class F1**, each class weighted **equally** — sensitive to rare classes. |
| **Weighted F1** | `f1_score(..., average="weighted")` | F1 averaged **by class support** (how many samples per class). |
| **Loss** | Mean batch **cross-entropy** over non-skipped batches | What AdamW minimizes. |

Batches with non-finite inputs/logits/loss are **skipped** and counted (`skipped_batches`).

### 4.3 What happens each epoch

- **Training:** `run_epoch` with **optimizer** → gradients, **gradient clipping** (`max_grad_norm`), **AdamW** step.  
- **Validation:** `run_epoch` with **`optimizer=None`** → no weight updates; **`val_macro_f1`** drives **ReduceLROnPlateau** (maximize validation macro-F1).

### 4.4 Final test evaluation (notebook)

After all candidate models are trained, the notebook sets:

- **`best_name`** = row with highest **`best_val_macro_f1`** in **`results_df`**  
- **`best_model`** = `trained_models[best_name]` with **best validation weights** restored  

Then it runs **`run_epoch(best_model, test_loader, ...)`** and prints **test loss**, **test macro F1**, and **test weighted F1**.

### 4.5 Per-city diagnostics (notebook)

For each city, test sequences for that city only: **macro F1**, **weighted F1**, and **accuracy = fraction of sequences where `pred == y_true`**.

---

## 5. What “accuracy” means here

- **Primary model-selection metric in the notebook:** **validation macro-F1** (not raw accuracy).  
- **Test reporting:** **macro-F1** and **weighted F1** are the headline numbers from `run_epoch` on the test loader.  
- **Accuracy** appears mainly in the **per-city** table: **correct argmax predictions / number of test sequences** for that city.  

**Globally**, “accuracy” on all test sequences is the same idea: **# correct predicted classes / # test windows**. It can be high when classes are easy or skewed; **macro-F1** is intentionally used to treat **all four risk levels** more fairly under imbalance.

The script **`src/evaluate_lstm.py`** also reports sklearn **accuracy** on the full test set for the **GRU** — same definition: **exact match** between predicted and true class index.

---

## 6. How the best model is chosen (and what counts as a baseline)

### 6.1 Deep learning “competition” inside the notebook

The notebook trains **four** architectures on the **same** data, splits, and batching:

| Model name | Type |
|------------|------|
| `GRU_Attn` | GRU + attention + city embedding + head |
| `LSTM_Attn` | **LSTM** + attention + city embedding + head |
| `TCN` | Temporal convolutional net + city embedding + head |
| `Transformer` | Transformer encoder + pooled sequence + city embedding + head |

For **each** model:

- Training tracks **validation macro-F1** each epoch.  
- The **weight snapshot** with the **highest validation macro-F1** is kept (`best_state`).  
- **Early stopping:** if validation macro-F1 does not improve for **`patience`** epochs (default **12**), training stops and weights reload to the best snapshot.

After training all four, **`results_df`** stores **`best_val_macro_f1`** per model and **sorts descending**. The **winner** is **`results_df.iloc[0]["model"]`** — whoever achieved the **best validation macro-F1**. This project **deploys the GRU** as **`gru_attn_best.pkl`** by default.

So the **baseline for choosing the DL forecaster** is: **the other deep architectures on identical data and metrics**, not a trivial “always predict class 0” rule.

### 6.2 Majority-class baseline (elsewhere)

**`src/evaluate_lstm.py`** compares the **GRU** to a **majority-class baseline** on the test set. That is useful for **reports**, not for picking the winner inside the notebook.

---

## 7. Comparison with the legacy (tabular) edition

The repo also has a **classical** forecasting path: **`HistGradientBoostingClassifier`** on **single-row** climate features (`forecast_hgb.pkl`, `feature_cols_forecast.pkl`), optionally invoked with **`FORECAST_USE_HGB=1`** when calling **`src/forecast.py`**. That is the natural **“legacy edition”** comparison to the **GRU + attention** pipeline.

| Topic | **GRU + attention (`gru_attn_best.pkl`)** | **Legacy HGB (and related sklearn)** |
|--------|---------------------------------------------|--------------------------------------|
| **Input shape** | **`(batch, seq_len, n_features)`** + **city id** | **One row** per (city, month): `feature_cols_forecast` |
| **Time modeling** | GRU **recurrent state** + **attention** over the window; lags may still appear **inside** `feature_cols` | Lags/rolls as **explicit columns** only |
| **City** | **Learned embedding** | No embedding; city is implicit in rows |
| **Algorithm** | PyTorch **GRU** (LSTM/TCN/Transformer only in the notebook competition) | **HistGradientBoosting** (or logreg etc. in tabular eval) |
| **Training loss** | Weighted **cross-entropy** + label smoothing | Boosting / logistic loss (sklearn) |
| **Selection criterion (DL notebook)** | **Best validation macro-F1** among DL candidates | Fixed HGB hyperparameters in `train.py` (separate pipeline) |
| **Artifact** | `.pkl` / `.pt` dict with weights + `feature_cols` + `city_to_idx` + `config` | `joblib` model + separate feature list pickle |
| **Forecast code** | **`forecast_lstm.py`** (recursive window + torch) | **`forecast.py`** `forecast_city` + `predict_proba` |

Both aim at **similar operational goals** (monthly risk under climate-only forecasting), but the **GRU** path is the one **selected** in **`deep_learning_model_selection.ipynb`** and packaged as **`gru_attn_best.pkl`** for dashboard forecasts (**`SEQUENCE_CHECKPOINT_NAME`**).

---

## Quick reference: notebook vs checkpoint

| Item | Notebook | Checkpoint (e.g. `gru_attn_best.pkl`) |
|------|----------|-----------------------------------|
| Training | All cells through model comparison + save | N/A |
| Best model | `results_df` sorted by `best_val_macro_f1` | Save/copy the winning run to `models/`; set **`SEQUENCE_CHECKPOINT_NAME`** if not using the default filename |
| Inference | Same `RNNAttentionClassifier` logic | `load_lstm_checkpoint` in `lstm_risk_model.py` |

---

*End of deep learning README. Main project instructions: [README.md](README.md).*