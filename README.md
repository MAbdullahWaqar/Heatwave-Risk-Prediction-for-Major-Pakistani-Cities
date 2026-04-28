# Urban Heat Stress Risk Forecasting — Pakistan (CS-245 Capstone)

This repository implements an end-to-end system to **forecast monthly urban heat stress risk** (four ordinal classes: **Low → Moderate → High → Extreme**) for major Pakistani cities. The **primary sequence model** is a **PyTorch GRU** sequence classifier with **learned attention over time** and **per-city embeddings** (default **`gru_attn_best.pkl`**). The **deep learning notebook** trains and exports the **GRU + attention** checkpoint used everywhere else.

The **notebook only reads the files declared in its `Config`** (see [Deep learning notebook — data and workflow](#deep-learning-notebook--data-and-workflow)); it does **not** load raw weather or World Bank CSVs directly. Those sources are incorporated earlier when **`df_model_forecast.csv`** is built via **`src/train.py`**. Evaluation and forecasting call the same **GRU** weights (`gru_attn_best.pkl` by default).

---

## Table of contents

1. [Repository layout](#repository-layout)  
2. [Problem and task definition](#problem-and-task-definition)  
3. [Data engineering pipeline (raw → processed)](#data-engineering-pipeline-raw--processed)  
4. [Heat Stress Index and risk labels](#heat-stress-index-and-risk-labels)  
5. [Forecast vs monitoring datasets](#forecast-vs-monitoring-datasets)  
6. [Deep learning notebook — data and workflow](#deep-learning-notebook--data-and-workflow)  
7. [Deep learning model: GRU + attention](#deep-learning-model-gru--attention)  
8. [Why “bidirectional” appears with the GRU](#why-bidirectional-appears-with-the-gru)  
9. [Sequence construction, training, and checkpoint](#sequence-construction-training-and-checkpoint)  
10. [Offline evaluation and explainability](#offline-evaluation-and-explainability)  
11. [Recursive forecasting (`forecast_lstm.py`)](#recursive-forecasting-forecast_lstm)  
12. [Feature design vs GRU core](#feature-design-vs-gru-core)  
13. [Scripts and modules reference](#scripts-and-modules-reference)  
14. [Outputs and artifacts](#outputs-and-artifacts)  
15. [Reproducible run order](#reproducible-run-order)  
16. [Streamlit dashboard](#streamlit-dashboard)  
17. [Troubleshooting](#troubleshooting)  
18. [Limitations and extensions](#limitations-and-extensions)  
19. [Author](#author)

---

## Repository layout

```
Heatwave-Risk-Prediction-for-Major-Pakistani-Cities/
├── requirements.txt                 # Python deps (repo root): torch, sklearn, streamlit, …
└── heat-risk-pk/
    ├── app/
    │   └── app.py                   # Streamlit UI (maps, timelines, scenarios, what-if)
    ├── data/
    │   ├── raw/                     # Source CSVs (see Data sources)
    │   └── processed/             # df_model_forecast.csv, df_model_monitoring.csv, optional merged weather
    ├── models/
    │   ├── gru_attn_best.pkl        # PyTorch **GRU** + attention (export from notebook; `SEQUENCE_CHECKPOINT_NAME`)
    │   ├── lstm_attn_best.pkl       # Optional alternate checkpoint only if you set SEQUENCE_CHECKPOINT_NAME
    │   ├── feature_cols_forecast.pkl   # Written by train.py (column list for GRU / merges)
    │   ├── feature_cols_monitoring.pkl # Monitoring feature list (optional analyses)
    │   └── metrics.json             # Written by train.py (pipeline metadata)
    ├── notebooks/
    │   ├── deep_learning_model_selection.ipynb  # **Train + export GRU** → `gru_attn_best.pkl`
    │   ├── data_processing.ipynb, Pakistani_data.ipynb, …  # Exploratory / ETL helpers
    │   └── ML_Pipeline_Complete.ipynb, MLProj.ipynb          # Historical ML walkthroughs
    ├── outputs/
    │   ├── forecasts/               # forecast_{6–72}m_{baseline|plus1c|plus2c}.csv
    │   └── figures/                 # GRU metrics, saliency, notebook SHAP exports, training curves
    └── src/
        ├── config.py                # Paths, splits, GRU `SEQUENCE_CHECKPOINT_NAME`
        ├── io.py                    # Load raw CSVs
        ├── preprocess.py            # Daily → monthly, city filtering
        ├── features.py              # WB long, surface temp PK, month sin/cos, climatology
        ├── targets.py               # HSI + risk_label
        ├── split.py                 # Lags, rollings, temporal_split()
        ├── train.py                 # build_dataset + save processed CSVs + feature column pickles (GRU pipeline)
        ├── merge_dl_features.py     # Humidity, NDVI, merged-weather joins (DL + forecast_lstm)
        ├── lstm_risk_model.py       # RNNAttentionClassifier, load_lstm_checkpoint
        ├── evaluate_lstm.py         # Test-set GRU eval + saliency
        ├── evaluate.py              # Entry → GRU eval (`evaluate_lstm`)
        ├── forecast.py              # Projection / HSI helpers; `main()` → GRU forecasts
        ├── forecast_lstm.py         # Recursive GRU scenario forecasts → CSV
        ├── generate_forecasts.py    # CLI wrapper calling forecast.main()
        ├── model_zoo.py             # Unused in default pipeline (kept for old notebooks)
        ├── explain.py               # Stub: use GRU saliency / notebook SHAP
        └── feature_importance.py    # Stub: use GRU saliency
```

---

## Problem and task definition

- **Geography**: Major Pakistani cities (coverage determined by raw weather data and `preprocess.filter_cities`).
- **Time grain**: **Month** (year, month index).
- **Target**: `risk_label` ∈ {0,1,2,3} derived from the **Heat Stress Index** (see below).
- **Operational goal**: Given **recent and current climate-related inputs**, estimate **probability over the four risk classes** for the **current** month in a forecasting setting, and **roll forward** under **warming / urbanization / population** scenarios for planning dashboards.

The **GRU** answers: *“Given the last `seq_len` months of features for this city, what is the risk distribution for the month at the end of that window?”* During **forecast simulation**, synthetic future months are appended so the window always ends on the month being projected.

---

## Data engineering pipeline (raw → processed)

This section describes how **`data/processed/df_model_forecast.csv`** is **produced** (e.g. `python -m src.train`). The **deep learning notebook does not open these raw inputs**; it only consumes the processed paths listed in [Deep learning notebook — data and workflow](#deep-learning-notebook--data-and-workflow).

| Step | Implementation | Description |
|------|----------------|-------------|
| 1 | `src/io.py` | Reads `pakistan_city_weather_daily.csv`, World Bank population density and urban % CSVs (skiprows=4), and `average-monthly-surface-temperature.csv` (PAK monthly surface temp). |
| 2 | `src/preprocess.daily_to_monthly` | Aggregates daily station weather to **city–year–month** means/sums (tavg, tmax, tmin, prcp, wspd, pres, tsun, n_days). |
| 3 | `src/preprocess.filter_cities` | Keeps cities with **≥ `MIN_MONTHS_PER_CITY` (600)** months and **≥ 80%** non-null `tavg_mean` (`src/config.py`). |
| 4 | `src/features.wb_to_long` | Melts World Bank wide tables to `(year, pop_density)` and `(year, urban_pct)` for **PAK**. |
| 5 | `src/features.surface_pk_monthly` | Extracts Pakistan surface temperature by year/month. |
| 6 | `src/features.add_time_features` | `month_sin`, `month_cos` (cyclical month encoding). |
| 7 | `src/features.add_city_climatology` | Per **(city, month)** mean `tavg_mean` → `tavg_clim`; anomaly `tavg_anom = tavg_mean - tavg_clim`. |
| 8 | `src/targets.add_heat_index` | Z-scores and **Heat Stress Index** (formula below). |
| 9 | `src/targets.add_risk_label` | Quantile bins on HSI using `P50`, `P75`, `P90` from `config.py` (0.5, 0.75, 0.9). |
| 10 | `src/split.add_lags_rollings` | Per city, sorted by time: `heat_lag_{1,3,6}`, `risk_lag_{1,3,6}`, rolling mean/std of heat (windows 3 and 6, shifted to avoid same-month leakage in rolls). |
| 11 | `src/train.get_feature_sets` | Builds **full** column set and **climate-only** set (drops `risk_lag_*` for forecasting table). |
| 12 | `src/train.train_models` | Writes `df_model_monitoring.csv`, `df_model_forecast.csv`, sklearn pickles, `metrics.json`. |

---

## Deep learning notebook — data and workflow

Everything below matches **`heat-risk-pk/notebooks/deep_learning_model_selection.ipynb`**: that notebook **only** loads and merges the datasets defined in its **`Config`** dataclass. No other files are read inside the notebook for training.

### Data files used by the notebook (and only these)

Paths are relative to the notebook folder (`heat-risk-pk/notebooks/`). In code they appear as `../data/...`.

| `Config` field | Path (from repo: `heat-risk-pk/…`) | Role |
|----------------|-------------------------------------|------|
| `data_path` | `data/processed/df_model_forecast.csv` | **Required.** Base monthly table: engineered features, `risk_label`, `city`, and either `date` or `year`/`month`. The **GRU** consumes **numeric columns derived from this table** (after merges). |
| `humidity_path` | `data/raw/pakistan_humidity_daily.csv` | **Optional.** If the file exists (`os.path.exists`), daily rows are aggregated to city–year–month means; columns such as `rh_avg`, `rh_max`, `rh_min`, `prcp`, `et0` (when present) are prefixed with `hum_` and left-joined. If missing, this step is skipped. |
| `ndvi_path` | `data/raw/pakistan_ndvi_monthly.csv` | **Optional.** If the file exists, NDVI is averaged to city–year–month as `ndvi_monthly` and left-joined. If missing, skipped. |
| `merged_weather_scaled_path` | `data/processed/pakistan_weather_merged_scaled.csv` | **Optional.** If the file exists and has `time` + `city`, selected columns are aggregated to monthly `wm_*` features and left-joined. If missing, skipped. |

After merges, the notebook builds **`feature_cols`** as all **numeric** columns except metadata, target, `year`/`month`, and **`risk_lag_*`** (explicitly excluded to avoid label leakage in forecasting-style learning).

**Where the base CSV comes from:** `df_model_forecast.csv` is **not** created inside this notebook. It is expected to **already exist** (typically produced by `src/train.py`, which merges weather, World Bank, surface temperature, etc., into the monthly pipeline described [above](#data-engineering-pipeline-raw--processed)). The notebook **treats that file as the single mandatory input table** and only adds the three optional joins above.

**Inference parity:** `src/merge_dl_features.py` replicates the same three optional merges so `forecast_lstm.py` and `evaluate_lstm.py` see the same column set as training.

### What the notebook does (model preparation and selection)

1. **Load** `df_model_forecast.csv` and construct `date` from `year`/`month` if needed.  
2. **Conditionally merge** humidity, NDVI, and merged-weather files when paths exist on disk.  
3. **Define features** (numeric, non-leakage columns) and **splits** by year: train ≤ 2015, val 2016–2019, test ≥ 2020 (same years as `Config.train_end_year` / `val_end_year`).  
4. **Impute** missing values with **training-set medians**; **fit `StandardScaler` on training rows only**, apply to all splits.  
5. **Build sequences** per city: sliding windows of length **`seq_len`** (default 12), target = `risk_label` at the end of the window.  
6. **Train** **GRU + Attention** with the supervised setup above (early stopping on **validation macro-F1**, class-weighted loss, AdamW, scheduler, gradient clipping — hyperparameters in `Config`).  
7. **Export** **`models/gru_attn_best.pkl`** (override with env **`SEQUENCE_CHECKPOINT_NAME`** only for experiments), including `model_state_dict`, `feature_cols`, `city_to_idx`, `config`, and `model_name` (`GRU_Attn`).

So: the notebook’s **declared data scope** is exactly the **one required processed CSV** plus **up to three optional files** named in `Config`. All other “sources” (daily weather, World Bank, …) are only relevant **upstream**, when building `df_model_forecast.csv`.

---

## Heat Stress Index and risk labels

**Z-scores** in `add_heat_index` use each column’s **mean and standard deviation over the dataframe passed into that function** at that step (column-wise, not per-city). Then:

```
heat_stress_index =
    0.40 * tavg_z
  + 0.30 * tmax_z
  + 0.15 * surf_z
  + 0.10 * pop_z
  + 0.10 * urb_z
  - 0.05 * wind_z
```

(`src/targets.py` — `wspd_mean` filled with median before `wind_z`.)

**Risk labels** (ordinal 0…3): compare HSI to **global** quantiles of HSI on the dataframe at binning time — `P50`, `P75`, `P90` (defaults 50th, 75th, 90th percentiles): Low / Moderate / High / Extreme.

---

## Forecast vs monitoring datasets

| File | Features | Use case |
|------|----------|----------|
| `data/processed/df_model_forecast.csv` | Climate / heat / lags of **heat** only — **no `risk_lag_*`** | **Forward forecasting** and **GRU training/eval/forecast** without observing future **risk** classes. |
| `data/processed/df_model_monitoring.csv` | Same + **`risk_lag_*`** (past observed risk) | **Optional** analyses when label history exists; the default **forecast** path uses the climate-only file below. |

Using `risk_lag_*` to predict the **current** month’s risk is useful for “where are we now?” dashboards with label history, but it is **not** the same problem as projecting **years ahead** where those lags are unknown.

---

## Deep learning model: GRU + attention

The production default is **`heat-risk-pk/models/gru_attn_best.pkl`**, produced by **`notebooks/deep_learning_model_selection.ipynb`** (**GRU + Attention**, early stopping on validation **macro-F1**). The **`RNNAttentionClassifier`** lives in **`src/lstm_risk_model.py`** and is used by `evaluate_lstm.py` and `forecast_lstm.py`.

### High-level architecture

1. **Input**: batch of shape **(N, T, F)** where **T = `seq_len`** (default **12** months), **F = len(feature_cols)** per checkpoint. A parallel tensor **city_idx** (N,) indexes the city embedding table.

2. **Recurrent core:** **`torch.nn.GRU`** with **`bidirectional=True`** — hidden states at **every** time step.

3. **Attention pooling**: `AttentionPool` scores each time step → **softmax over time** → **weighted sum** of GRU outputs → one **context vector** per sample.

4. **City conditioning**: `nn.Embedding(num_cities, embed_dim)` (default **embed_dim = 8**); concatenated with the context vector.

5. **Classifier head**: `Linear → ReLU → Dropout → Linear` → **4 logits** → **softmax** for class probabilities. Training uses **cross-entropy** (with optional class weights in the notebook).

### Implementation constants (defaults in code / notebook)

| Component | Typical value | Where defined |
|-----------|----------------|---------------|
| RNN type | **GRU** (`nn.GRU`) | `gru_attn_best.pkl`; notebook `GRU_Attn` |
| `bidirectional` | **True** | `lstm_risk_model.py` |
| `hidden_dim` | **64** | Notebook `Config`; checkpoint `config` |
| `num_layers` | **2** | Same |
| `dropout` | **0.25** | GRU inter-layer + MLP head |
| `seq_len` | **12** | Notebook `Config`; `forecast_lstm` / `evaluate_lstm` read from checkpoint |
| `embed_dim` | **8** | City embedding |
| Optimizer | **AdamW**, lr **3e-4**, weight decay **1e-4** | Notebook `Config` |
| Max epochs | **80** | Notebook |
| Early stopping patience | **12** | Notebook |
| LR scheduler | **ReduceLROnPlateau** (maximize val score, factor 0.5, patience 4) | Notebook |
| Gradient clipping | **max_norm 1.0** | Notebook `max_grad_norm` |
| Batch size | **32** | Notebook |

Forecasts carry **`model_name` `GRU_Attn`** from this checkpoint.

---

## Why “bidirectional” appears with the GRU

The **`nn.GRU`** is constructed with **`bidirectional=True`**: two independent chains over the **fixed length-T context window**, **concatenated** per step so attention and the MLP use dimension **2 × hidden_dim**.

In **`src/lstm_risk_model.py`**, the GRU uses the same bidirectional pattern as in the notebook. **Bidirectional** here means forward and backward over **months inside the window**, not future **labels** — the label is always for the **last** month of the window.

The **label** in supervised learning is still for the **last month** in each window (the sequence ends at the prediction month). Bidirectionality does **not** use future **labels**; it only uses future **months inside the feature window** relative to earlier steps in that same window (standard for **encoding** a finite context segment).

---

## Sequence construction, training, and checkpoint

**Per city**, rows are sorted by date. For each index `t` from `seq_len-1` to `len-1`, one training sample is:

- `X`: features for months `[t-seq_len+1, …, t]` → shape `(seq_len, F)`
- `y`: `risk_label` at month `t`
- `city_idx`: city id at month `t`

**Splits** (notebook and `evaluate_lstm.py`): by **calendar year** — train ≤ `TRAIN_END_YEAR` (**2015**), validation **2016–2019** (`VAL_END_YEAR`), test **≥ 2020** (`src/config.py`).

**Preprocessing (leakage-safe)**:

- Numeric coercion, inf → NaN.
- **Median imputation** using **training rows only** for each feature column.
- **`StandardScaler`** fit on **training rows only**, applied to all data and reused in `forecast_lstm.py` on history + simulated futures.

**Checkpoint contents** (typical): `model_state_dict`, `config` (hyperparameters), `feature_cols` (ordered list), `city_to_idx` (string → int), `model_name` (e.g. **`GRU_Attn`** for the default GRU weights file).

---

## Offline evaluation and explainability

- **Command**: `cd heat-risk-pk && python -m src.evaluate` → runs **`src/evaluate_lstm.py`**.

**Metrics** (sklearn metrics on test **sequences**): accuracy, macro precision/recall/F1, **Extreme-class recall**, confusion matrix, classification report. Rows include **majority-class baseline** and **GRU** (`model_metrics.csv`).

**Neural attribution**: **Input × gradient** saliency (`sequence_feature_saliency_top15.png`, `sequence_feature_saliency.csv`) for the **GRU**.

**Kernel SHAP** on the torch model: optional cells in **`notebooks/deep_learning_model_selection.ipynb`** (not `src/explain.py`, which is a stub in the GRU-only pipeline).

---

## Recursive forecasting (`forecast_lstm.py`)

- Loads **`models/<SEQUENCE_CHECKPOINT_NAME>`** (default **`gru_attn_best.pkl`**), `merge_auxiliary_features`, rebuilds scaler on train years, builds **projection lookups** (`build_projection_lookups` in `forecast.py`) for climatology and urban/pop trends.
- For each city, initializes a **sequence buffer** from the tail of **observed** merged history.
- For each future month in the horizon: applies **scenario** adjustments (temperature delta, urban delta, population multiplier), updates **synthetic** climate fields and **heat stress index**, builds one new feature row, slides the window, runs **GRU → softmax**, appends CSV row with `pred_risk`, `p_low`…`p_extreme`.

**Outputs**: `outputs/forecasts/forecast_{6,12,24,36,48,60,72}m_{baseline,plus1c,plus2c}.csv` (exact set defined in `run_lstm_main`).

**Entry points**: `python -m src.forecast` (GRU from `config.py`), or `python src/generate_forecasts.py` from `heat-risk-pk/`.

---

## Feature design vs GRU core

Engineered **tabular** columns (lags, rolls, climate fields) are built in **`src/train.py`** and consumed as a **sequence window** by the **GRU**. The **risk classifier** is always the sequence **GRU + attention** checkpoint in **`forecast_lstm.py`** / **`evaluate_lstm.py`**.

---

## Scripts and modules reference

| Path | Role |
|------|------|
| `src/train.py` | `build_dataset()` → processed CSVs + feature column pickles + `metrics.json` (no sklearn risk models). |
| `notebooks/deep_learning_model_selection.ipynb` | Train **GRU + attention**; **export** `models/gru_attn_best.pkl`. |
| `src/merge_dl_features.py` | Join humidity, NDVI, optional merged weather onto a city–month dataframe. |
| `src/lstm_risk_model.py` | **`RNNAttentionClassifier`**: **GRU** + attention + head; `load_lstm_checkpoint` (can load non-GRU checkpoints only if you point **`SEQUENCE_CHECKPOINT_NAME`** at them). |
| `src/evaluate_lstm.py` | GRU test evaluation + saliency. |
| `src/evaluate.py` | Thin entry → **`evaluate_lstm`** (GRU test metrics + saliency). |
| `src/forecast_lstm.py` | `run_lstm_main`, `forecast_city_lstm` — GRU scenario CSV generation. |
| `src/forecast.py` | `compute_heat_index`, `build_projection_lookups`, `next_year_month`; `main()` → GRU forecaster. |
| `src/generate_forecasts.py` | Invokes `forecast.main()` with path fix for script execution. |
| `app/app.py` | Streamlit application. |

---

## Outputs and artifacts

| Location | Contents |
|----------|----------|
| `outputs/forecasts/*.csv` | Precomputed **GRU** forecasts; columns include `forecast_model`, `forecast_checkpoint`, scenario knobs, probabilities. |
| `outputs/figures/model_metrics.csv` | After `python -m src.evaluate`: GRU vs majority baseline. |
| `outputs/figures/confusion_matrix_sequence.png` | **GRU** test confusion matrix. |
| `outputs/figures/sequence_feature_saliency*.png,.csv` | Gradient-based importance for **GRU** inputs. |
| `outputs/figures/shap_*.csv` / SHAP plots | From the **notebook** Kernel SHAP cells (optional). |

---

## Reproducible run order

```bash
# 1) Environment (from repo root)
python -m venv .venv && source .venv/activate
pip install -r requirements.txt

# 2) Build processed tables + feature column pickles (no sklearn forecaster)
cd heat-risk-pk && python -m src.train

# 3) Jupyter: heat-risk-pk/notebooks/deep_learning_model_selection.ipynb
#    Train GRU + attention; early stop on val macro-F1; saves models/gru_attn_best.pkl

# 4) Evaluate GRU checkpoint on held-out years
cd heat-risk-pk && python -m src.evaluate

# 5) Generate forecast CSVs (GRU)
cd heat-risk-pk && python -m src.forecast
# or:  cd heat-risk-pk && python src/generate_forecasts.py

# 6) Dashboard (from repo root)
streamlit run heat-risk-pk/app/app.py
```

---

## Streamlit dashboard

- Reads **`outputs/forecasts/`** and **`outputs/figures/`**.
- **Map**, **city timelines**, **scenario comparison** (baseline / +1°C / +2°C style files), **what-if** (**Predict Now** then sliders — heuristic rescoring of saved GRU probabilities, not online learning).
- Displays **GRU** metrics, confusion matrix, and saliency; optional notebook SHAP CSVs if you generated them.

---

## Troubleshooting

| Issue | Action |
|-------|--------|
| Missing forecasts | `cd heat-risk-pk && python src/generate_forecasts.py` |
| `No module named 'src'` | Run commands with **`cd heat-risk-pk`** first, or `python -m src.<module>`. |
| Missing checkpoint | Complete notebook training and place the chosen weights under `heat-risk-pk/models/` (default **`gru_attn_best.pkl`**; set **`SEQUENCE_CHECKPOINT_NAME`** if using another filename). |
| Empty sequences in eval | Ensure processed CSV spans years beyond `VAL_END_YEAR` and cities have ≥ `seq_len` test months. |

---

## Limitations and extensions

- **Recursive** scenario forecasts with **climatology-style** inputs tend to **repeat seasonal patterns** at long horizons; interpret as **month-of-year risk** and **relative scenarios**, not precise far-future year ordering.
- Risk classes are **percentile-based** on historical HSI, not health-outcome calibrated thresholds.
- Extensions: CMIP6 forcings, explicit **year** or trend features, probabilistic calibration, larger architectures or ensembles.

---

## Author

- Muhammad Abdullah Waqar

Supplementary PDF (if included in your copy): `heat-risk-pk/ML_Project_Final_Report.pdf`.
