# Urban Heat Stress Risk Forecasting for Major Pakistani Cities

End-to-end **deep learning** pipeline for **monthly ordinal heat-risk** (Low → Extreme), **12-month sequences**, **scenario forecasts**, and a **Streamlit** dashboard. See **Table of contents** below.

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Problem, motivation, and operational framing](#2-problem-motivation-and-operational-framing)
3. [Learning objectives and evidence of experimentation](#3-learning-objectives-and-evidence-of-experimentation)
4. [Data sources, quality filters, and engineering pipeline](#4-data-sources-quality-filters-and-engineering-pipeline)
5. [Heat Stress Index (HSI) and ordinal risk labels](#5-heat-stress-index-hsi-and-ordinal-risk-labels)
6. [Feature design, lag structure, and leakage controls](#6-feature-design-lag-structure-and-leakage-controls)
7. [Temporal splits and evaluation protocol](#7-temporal-splits-and-evaluation-protocol)
8. [Sequence formulation for deep learning](#8-sequence-formulation-for-deep-learning)
9. [Model zoo: architectures, inductive biases, and outcomes](#9-model-zoo-architectures-inductive-biases-and-outcomes)
10. [Training objective, optimization, and regularization](#10-training-objective-optimization-and-regularization)
11. [Results: held-out metrics, confusion analysis, and baselines](#11-results-held-out-metrics-confusion-analysis-and-baselines)
12. [Explainability: saliency, SHAP, and narrative alignment](#12-explainability-saliency-shap-and-narrative-alignment)
13. [Forecasting engine, scenarios, and semantic caveats](#13-forecasting-engine-scenarios-and-semantic-caveats)
14. [Streamlit dashboard: capabilities and dependencies](#14-streamlit-dashboard-capabilities-and-dependencies)
15. [How to run: environment, commands, and troubleshooting](#15-how-to-run-environment-commands-and-troubleshooting)
16. [Repository layout and artifact contracts](#16-repository-layout-and-artifact-contracts)
17. [Impact pathways, ethics, and deployment conditions](#17-impact-pathways-ethics-and-deployment-conditions)
18. [Insights, limitations, and future work](#18-insights-limitations-and-future-work)
19. [FAQ](#19-faq)
20. [Glossary](#20-glossary)
21. [Appendix A — Extended design decision log](#appendix-a--extended-design-decision-log)
22. [Appendix B — Output artifacts reference](#appendix-b--output-artifacts-reference)
23. [Appendix C — Conceptual viva preparation](#appendix-c--conceptual-viva-preparation)
24. [Authoring and citation notes](#authoring-and-citation-notes)

---

## 1. Executive summary

Urban heat is a **compound risk**: temperature extremes interact with **urbanization**, **population exposure**, and **background regional warming**. This project implements a **transparent monthly panel** at **city–month** resolution, constructs a **Heat Stress Index (HSI)** from z-scored drivers, maps HSI to **four ordinal risk classes** using **global quantile thresholds**, and trains **PyTorch** sequence classifiers that read **twelve consecutive months** of engineered features to predict the **risk label of the final month**.

The **best-validated architecture** in the project’s experiment hub is **GRU + attention**, which achieves strong **macro-F1** and especially strong **Extreme-class recall** on **held-out test years** compared to a **majority-class baseline** that collapses under imbalance. The system exports **checkpoints**, **metrics**, **confusion matrices**, **saliency/SHAP artifacts**, **forecast CSVs** under stylized warming scenarios, and a **Streamlit** UI for communication.

**What this is not:** It is not a replacement for **numerical weather prediction** or **seasonal forecasting models** operated by meteorological agencies. It is a **decision-support and communication layer** that turns heterogeneous observational panels into **repeatable risk tiers** and **scenario contrasts**.

---

## 2. Problem, motivation, and operational framing

### 2.1 Real-world context (Pakistan)

Pakistan experiences recurrent **extreme heat**, urban **heat island** effects, and **compound stressors** (dense settlement, energy access patterns, reduced nocturnal cooling). Municipal actors, public health departments, and disaster management authorities increasingly need **structured indicators** that are:

- **Repeatable** (same inputs → same outputs),
- **Auditable** (weights and thresholds are explicit),
- **Communicable** to non-technical stakeholders,
- **Scenario-aware** for “what if it gets warmer or more urban?” discussions.

### 2.2 Problem statement (technical)

Given a **history of monthly engineered features** for a city, estimate a **probability distribution** over **four ordered risk classes** for the **current month** at the end of the history window, where classes correspond to **HSI quantile bins** computed on the modeling frame.

Additionally, produce **forward-looking scenario projections** by recursively synthesizing future feature rows under **baseline**, **+1 °C**, and **+2 °C** stylized temperature adjustments (plus optional urban/population knobs), feeding the **sequence model** at each step to generate **risk trajectories** suitable for **comparative** dashboards.

### 2.3 Operational goal

Enable **prioritization** and **risk communication**: which cities and months show elevated **Extreme** probabilities under stylized warming, and how do **class probabilities** shift relative to baseline? The goal is **analysis and interpretation**, not a single scalar “heat score” without context.

### 2.4 Why deep learning for this panel?

The mapping from **seasonal context**, **multi-month memory**, and **city-specific baselines** to ordinal classes is **nonlinear** and **highly interactive**:

- The same absolute temperature can imply different stress depending on **monthly climatology** and **anomaly** state.
- **Rolling heat statistics** encode persistence; **lags** encode memory.
- **City embeddings** capture systematic residuals.

 Classical **tabular models** can perform well, but the capstone explicitly demonstrates a **PyTorch sequence pipeline** with **multiple architectures** and **rigorous held-out evaluation** plus **deployment artifacts** (checkpoint consumption in evaluation, forecasting, and UI).

---

## 3. Learning objectives and evidence of experimentation

### 3.1 Demonstrated competencies

- **Working deep learning solution** end-to-end: training (notebook), export, `python -m src.evaluate`, `python -m src.forecast`, `streamlit run ...`.
- **Non-trivial problem structure**: multi-city panels, class imbalance, temporal dependence, optional sparse merges.
- **Experimentation**: GRU, LSTM, TCN, Transformer compared under shared preprocessing; **negative result** documented for Transformer.
- **Thoughtful metrics**: accuracy **and** macro-F1 **and** Extreme recall **and** confusion matrices.
- **Interpretability**: saliency + SHAP artifacts as complementary lenses.

### 3.2 Why “no single correct solution” matters

Label definition uses **quantile cuts** on an **expert-weighted index**—reasonable, but not unique. Architecture choice is not unique. Split strategy trades **bias vs variance**. The writeup highlights **trade-offs** rather than pretending a single pipeline is universally optimal.

---

## 4. Data sources, quality filters, and engineering pipeline

### 4.1 Raw inputs (typical)

Located under `heat-risk-pk/data/raw/`:

- **Daily city weather** (`pakistan_city_weather_daily.csv`): station-based daily aggregates used to build monthly means/sums.
- **World Bank** population density (`API_EN.POP.DNST_DS2_en_csv_v2_110190.csv`) and urban population share (`API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_110318.csv`), filtered to **PAK**.
- **Monthly surface temperature** (`average-monthly-surface-temperature.csv`) filtered to **PAK** rows.
- **Optional**: humidity daily (`pakistan_humidity_daily.csv`), NDVI monthly (`pakistan_ndvi_monthly.csv`), merged scaled weather (`data/processed/pakistan_weather_merged_scaled.csv`) for notebook/eval parity.

### 4.2 Aggregation and filtering (`src/preprocess.py`, `src/config.py`)

- Daily weather is parsed to datetimes and aggregated to **city–year–month** with grouped means/max/sums and day counts.
- Cities are retained only if:
  - `n_months >= MIN_MONTHS_PER_CITY` (**600** months), and
  - average non-null fraction of `tavg_mean` ≥ `MIN_TAVG_NONNULL` (**0.80**).

**Rationale:** short histories break twelve-month windows and make climatology unstable; sparse temperature series makes z-scored indices untrustworthy.

### 4.3 Merges and feature enrichment (`src/train.py`, `src/features.py`)

- World Bank indicators are melted to long annual tables and merged on **year**.
- Surface temperature is parsed to **year–month** and merged on **year, month**.
- Cyclical month features (`month_sin`, `month_cos`) are added.
- City-month climatology `tavg_clim` is computed from grouped means; `tavg_anom` is deviation from climatology.

### 4.4 Targets, lags, and processed export (`src/targets.py`, `src/split.py`, `src/train.py`)

- Rows missing key columns for HSI are dropped (`tavg_mean`, `tmax_mean`, `pop_density`, `urban_pct`, `surface_temp_avg`).
- HSI and `risk_label` are computed.
- Lags and rolling statistics are added (`heat_lag_*`, `risk_lag_*`, `heat_roll_*`), with rolling operations **shifted** to reduce improper same-month leakage from rolling means/std.

**Forecast modeling column list** (`get_gru_feature_set`) is intentionally **strict** and **does not include** `risk_lag_*` features, avoiding a **trivial label leakage path** in deployment-style inference.

**Artifacts written:**

- `heat-risk-pk/data/processed/df_model_forecast.csv`
- `heat-risk-pk/models/feature_cols_forecast.pkl`
- `heat-risk-pk/models/metrics.json`

---

## 5. Heat Stress Index (HSI) and ordinal risk labels

### 5.1 HSI definition (implemented)

Column z-scores are computed within the frame at HSI construction time in `add_heat_index`:

\[
\text{HSI} = 0.40\,z_{\text{tavg}} + 0.30\,z_{\text{tmax}} + 0.15\,z_{\text{surf}} + 0.10\,z_{\text{pop}} + 0.10\,z_{\text{urb}} - 0.05\,z_{\text{wind}}
\]

Wind speed is **median-filled** before z-scoring to reduce sensitivity to missing anemometer stretches.

### 5.2 Ordinal mapping

`add_risk_label` applies **global quantiles** of HSI with defaults `P50=0.50`, `P75=0.75`, `P90=0.90`:

- ≤ p50 → **Low (0)**
- ≤ p75 → **Moderate (1)**
- ≤ p90 → **High (2)**
- else → **Extreme (3)**

**Rationale:** ordinal tiers are **governance-friendly** and stable under modest distributional shifts, compared to per-city quantiles that would desynchronize cross-city comparability.

**Caveat:** tiers are **not** calibrated to **mortality** or **morbidity**; they are **relative stress** strata for modeling and communication unless externally calibrated.

---

## 6. Feature design, lag structure, and leakage controls

### 6.1 Core features used by the default GRU table (`get_gru_feature_set`)

- `tavg_mean`, `tmax_mean`, `tavg_anom`
- `surface_temp_avg`
- `pop_density`, `urban_pct`
- `month_sin`, `month_cos`
- `heat_lag_1`, `heat_lag_3`, `heat_lag_6`
- `heat_roll_mean_3`, `heat_roll_std_3`, `heat_roll_mean_6`, `heat_roll_std_6`

### 6.2 Leakage philosophy

- Rolling statistics use a **shift** before rolling (`shift(1).rolling(w)`), preventing the current month’s target from incorporating **same-row smoothed future/present artifacts** inconsistent with causal deployment assumptions.
- **Risk lags** exist for analysis elsewhere but are excluded from the exported GRU feature list by design.

---

## 7. Temporal splits and evaluation protocol

Configured in `src/config.py`:

- **Train:** `year <= TRAIN_END_YEAR` (**2015**)
- **Validation:** `2016–2019` inclusive
- **Test:** `year > VAL_END_YEAR` (**≥ 2020**)

**Evaluation (`src/evaluate_lstm.py`)** fits **median imputation** and **StandardScaler** statistics using **train mask only**, then builds **sequences** for the **test** subset, compares model predictions to labels, and also evaluates a **majority-class baseline** fit on train+val labels.

**Why this split:** reduces **optimistic bias** from random splitting within cities, where adjacent months correlate strongly.

---

## 8. Sequence formulation for deep learning

For each city, rows are sorted by date. For each end index `t` with sufficient history, input is `X[t-11:t+1]` (twelve months) of `feature_cols`, city index corresponds to month `t`, target is `risk_label[t]`.

**Implication:** the model sees a **full annual context** leading into the labeled month, matching planning cycles and seasonal memory effects.

---

## 9. Model zoo: architectures, inductive biases, and outcomes

Trained in `heat-risk-pk/notebooks/deep_learning_model_selection.ipynb` under shared preprocessing:

| Model | Role | Validation macro-F1 (representative full run) |
|------:|------|-----------------------------------------------:|
| **GRU_Attn** | **Production default** | **0.907** |
| LSTM_Attn | Strong RNN alternative | 0.880 |
| TCN | Dilated convolutions, parallel inductive bias | 0.872 |
| Transformer | Deliberate stress test / negative result | 0.287 |

**Interpretation:** GRU edges LSTM slightly—consistent with **fewer parameters/gates** and good optimization on **moderate-size panels**. TCN remains competitive. Transformer underperforms here: a **useful negative result** illustrating **data scale**, **inductive bias**, and **tuning** matter more than architecture popularity.

**Deployment pointer:** `SEQUENCE_CHECKPOINT_NAME` defaults to `gru_attn_best.pkl` in `src/config.py`.

---

## 10. Training objective, optimization, and regularization

**Objective:** multi-class cross-entropy with **class weights** (imbalance control).

**Selection metric:** **validation macro-F1** for early stopping.

**Optimizer:** AdamW with weight decay; **ReduceLROnPlateau**; **gradient clipping**.

**Hyperparameters (notebook defaults, representative):** `hidden_dim=64`, `num_layers=2`, `dropout=0.25`, `embed_dim=8`, `batch_size=32`, `lr=3e-4`, `weight_decay=1e-4`, up to **80** epochs, patience **12**.

**Rationale:** macro-F1 aligns optimization with **balanced** performance; class weights fight **majority collapse**.

---

## 11. Results: held-out metrics, confusion analysis, and baselines

### 11.1 Sequence model (GRU + attention) — test sequences

From `heat-risk-pk/outputs/figures/best_model.txt` and `classification_report_sequence.txt` (n=**247** test windows):

| Metric | Value |
|------:|------:|
| Accuracy | 0.903 |
| Macro F1 | 0.887 |
| Macro precision | 0.890 |
| Macro recall | 0.888 |
| Extreme recall | 0.922 |

Per-class F1 (approximate): Low **0.97**, Moderate **0.83**, High **0.81**, Extreme **0.94**.

### 11.2 Majority baseline

From `classification_report_baseline.txt`:

- Accuracy ≈ **0.39**, macro F1 ≈ **0.14**
- Always predicts the majority class → **zero** precision/recall for non-majority classes

**Interpretation:** class imbalance makes accuracy misleading; the sequence model learns **non-trivial temporal structure** and **tail behavior**.

### 11.3 Consolidated CSV

`heat-risk-pk/outputs/figures/model_metrics.csv` contains side-by-side rows for `baseline_majority` and `GRU_Attn`.

---

## 12. Explainability: saliency, SHAP, and narrative alignment

- **Input × gradient saliency** (`sequence_feature_saliency*.png/csv`) highlights drivers that most influence the selected logits on a **test subsample**.
- **Kernel SHAP** (notebook-driven) exports `shap_feature_importance_best_model.csv`, where pressure/wind/humidity/seasonality/rolling heat fields appear among top contributors—consistent with **physical intuition**, though SHAP rankings are **not causal proof**.

---

## 13. Forecasting engine, scenarios, and semantic caveats

`src/forecast_lstm.py` loads the checkpoint, maintains a **twelve-month buffer**, constructs next-month feature rows using **climatology + anomaly medians + scenario deltas**, updates **HSI-like** pathways, and writes CSVs under `heat-risk-pk/outputs/forecasts/` for horizons such as **6–72** months and scenarios `baseline`, `plus1c`, `plus2c`.

**Caveat:** long recursive horizons are best read as **scenario stress** and **seasonal structure under knobs**, not precise calendar forecasts.

---

## 14. Streamlit dashboard: capabilities and dependencies

`streamlit run heat-risk-pk/app/app.py` loads precomputed **forecast CSVs** and **figures** to visualize timelines, maps (PyDeck), and model metrics cards. If forecasts are missing, the app surfaces a **clear run instruction** (`python -m src.forecast` from `heat-risk-pk`).

---

## 15. How to run: environment, commands, and troubleshooting

```bash
# Repository root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build processed dataset + feature list
cd heat-risk-pk && python -m src.train

# Train/compare/export in Jupyter
# Open heat-risk-pk/notebooks/deep_learning_model_selection.ipynb
# Export: heat-risk-pk/models/gru_attn_best.pkl

# Evaluate on held-out test years
cd heat-risk-pk && python -m src.evaluate

# Forecast CSVs
cd heat-risk-pk && python -m src.forecast

# Dashboard
streamlit run heat-risk-pk/app/app.py
```

**Module path rule:** run `python -m src.*` **from `heat-risk-pk`** so imports resolve.

**Checkpoint override:** `export SEQUENCE_CHECKPOINT_NAME=your.pkl` for experiments.

---

## 16. Repository layout and artifact contracts

```
Heatwave-Risk-Prediction-for-Major-Pakistani-Cities/
├── requirements.txt
├── README.md
└── heat-risk-pk/
    ├── app/app.py
    ├── data/raw/
    ├── data/processed/
    ├── models/
    ├── notebooks/
    ├── outputs/forecasts/
    ├── outputs/figures/
    └── src/
        ├── config.py
        ├── io.py
        ├── preprocess.py
        ├── features.py
        ├── targets.py
        ├── split.py
        ├── train.py
        ├── merge_dl_features.py
        ├── lstm_risk_model.py
        ├── evaluate.py
        ├── evaluate_lstm.py
        ├── forecast.py
        ├── forecast_lstm.py
        ├── generate_forecasts.py
        └── generate_shap.py
```

---

## 17. Impact pathways, ethics, and deployment conditions

**Potential impact**

- Heat action planning prioritization under stylized warming.
- Educational demonstration of responsible ML: metrics, baselines, limitations.
- Template for other **ordinal environmental risk** panels.

**Conditions**

- Validate against **local health/outcome** data before operational triggers.
- Replace stylized deltas with **climate projection ensembles** when available.
- Maintain **data provenance** and **versioning**.

---

## 18. Insights, limitations, and future work

- **Ordinal loss upgrades:** CORAL, cumulative link models, or Poisson-binomial structures.
- **Calibration:** temperature scaling / isotonic regression on softmax outputs.
- **Uncertainty:** ensembles, MC dropout, conformal sets.
- **Spatial models:** graph convolutions across cities with explicit edges.
- **Higher resolution:** sub-city heat islands if data exists.

---

## 19. FAQ

**Q1. Is this a weather forecast?**  
No. It scores **monthly risk tiers** from historical engineered panels and scenario projections.

**Q2. Why GRU over LSTM?**  
Empirical validation macro-F1 favored GRU here; LSTM remained competitive.

**Q3. Why did Transformer fail?**  
Likely **small effective sample**, **architecture mismatch**, and insufficient regularization/tuning for a tabular sequence derived from monthly rows.

**Q4. Can I add new cities?**  
You need **retraining** to learn embeddings and dynamics; unknown cities are dropped in current evaluation.

**Q5. Is Extreme risk a heatwave definition?**  
It is the **top decile tail** of the chosen HSI within the modeling distribution unless recalibrated.

**Q6. How do I reproduce figures?**  
Run `python -m src.evaluate` after training export and processed CSV exist.

**Q7. What if I lack humidity/NDVI files?**  
Merges are optional; base pipeline still runs.

**Q8. Why macro-F1?**  
To reflect performance on **rare** classes, not only the dominant Low months.

**Q9. Are probabilities calibrated?**  
Not guaranteed; treat softmax outputs as **scores** unless calibrated.

**Q10. Can I use CPU only?**  
Yes; evaluation and forecasting run on CPU by default if CUDA absent.

---

## 20. Glossary

- **HSI:** Heat Stress Index, weighted z-score composite.
- **HSI quantile labels:** Ordinal bins from global HSI quantiles.
- **Macro-F1:** Unweighted mean of class F1 scores.
- **Sequence window:** Twelve consecutive city-month rows.
- **Scenario forecast:** Recursive multi-step projection under knobs.
- **Majority baseline:** Always predict most frequent training label.

---

## Appendix A — Extended design decision log

The appendix enumerates **hundreds** of concise decision records (**DD-###**). Each repeats the core theme: *transparent defaults, leakage-aware features, honest baselines, and deployment-conscious preprocessing*.


### DD-001 — Temporal leakage and rolling windows (extended note)

Rolling means use a one-month shift so the current month's target does not leak smoothed information from the same calendar month in an improper way.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-002 — Year-based holdout versus random windows (extended note)

Random windows inside a city would let adjacent months appear in both train and test, inflating scores; chronological splits respect causality.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-003 — Macro-F1 versus accuracy for early stopping (extended note)

Accuracy is dominated by the majority class; macro-F1 forces the optimizer to care about rare Moderate/High/Extreme months.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-004 — Class-weighted cross-entropy (extended note)

Inverse-frequency or balanced weights reduce collapse to always predicting Low, which a naive baseline exhibits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-005 — Twelve-month context length (extended note)

Covers a full seasonal cycle while keeping the tensor small enough for limited data and stable optimization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-006 — Bidirectional RNN over the window (extended note)

The operational question is state at the end of a known segment; both directions summarize intra-window dynamics without using future labels.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-007 — Attention pooling over time (extended note)

Produces a weighted summary of which months in the year contributed most, aiding interpretability versus last-step pooling only.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-008 — City embeddings (extended note)

Cities differ in baseline climate and urban form; embeddings absorb residual city-specific effects not fully encoded in tabular columns.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-009 — GRU versus LSTM inductive bias (extended note)

GRU has fewer gates and often trains faster on modest panels; empirically it edged LSTM on validation macro-F1 here.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-010 — TCN as a convolutional alternative (extended note)

Parallel receptive fields over dilated convolutions stress a complementary inductive bias; results were competitive but slightly below GRU.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-011 — Transformer as a stress test (extended note)

Global mixing with limited data produced a negative result, illustrating that architecture hype must meet data scale and tuning.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-012 — AdamW and weight decay (extended note)

Decoupled weight decay regularizes large weights in the MLP head and embeddings without destabilizing momentum estimates.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-013 — ReduceLROnPlateau (extended note)

Lowering the learning rate when validation macro-F1 stalls helps late-stage fine-tuning without manual schedules.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-014 — Gradient clipping (extended note)

Recurrent stacks can explode gradients on heterogeneous city-month panels; clipping stabilizes training.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-015 — Median imputation fit on train only (extended note)

Using validation or test medians would leak information; train-only medians mirror deployment when new cities appear.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-016 — StandardScaler fit on train years only (extended note)

Same rationale as imputation: test statistics must never define the scaling used for test inputs.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-017 — Quantile thresholds for ordinal labels (extended note)

Global P50/P75/P90 splits on HSI yield four ordered tiers that are easy to audit and recompute if weights change.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-018 — HSI as expert-weighted composite (extended note)

Transparency for stakeholders beats opaque end-to-end label learning when health outcomes are not yet linked.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-019 — Wind enters with a negative coefficient (extended note)

Higher wind speeds modestly reduce composite stress in this simplified index; missing wind is median-filled before z-scoring.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-020 — Monthly aggregation from daily weather (extended note)

Means and sums summarize station behavior while controlling noise versus keeping every daily row.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-021 — Minimum months per city filter (extended note)

Cities with fewer than six hundred months are dropped to ensure long histories for sequence construction and robust climatology.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-022 — Minimum non-null tavg coverage (extended note)

Eighty percent coverage avoids cities where intermittent reporting would make z-scores and lags unreliable.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-023 — World Bank national drivers (extended note)

Population density and urban share are annual national proxies; coarse but consistent across cities for slow-changing exposure.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-024 — Surface temperature merge (extended note)

Regional-scale warming signal complements station means and anchors cross-city coherence in hot seasons.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-025 — Cyclical month encoding (extended note)

Sine and cosine preserve circular continuity of January and December unlike raw integer month.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-026 — Climatology and anomaly decomposition (extended note)

Separates expected seasonal warmth from deviations that often precede heat-stress escalations.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-027 — Recursive forecasting semantics (extended note)

Long horizons repeat seasonal structure under scenario knobs; interpret as stylized stress tests, not exact calendar futures.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-028 — Scenario temperature deltas (extended note)

Plus one and plus two degrees Celsius are simple, communicative warming knobs for comparative dashboards.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-029 — Streamlit for operations (extended note)

Rapid iteration for non-engineer stakeholders to explore precomputed forecasts without a heavy web stack.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-030 — Checkpoint payload structure (extended note)

Torch pickles bundle weights, config, feature list, and city index map so evaluation and forecast scripts stay in sync.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-031 — Majority baseline honesty (extended note)

Reporting a majority-class baseline shows class imbalance severity and justifies balanced metrics and weighted loss.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-032 — Per-city evaluation CSV (extended note)

Downstream UI and analysts can see heterogeneity: some cities have harder extreme-event detection than others.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-033 — Temporal error patterns by season bucket (extended note)

Aggregating errors by monsoon versus peak summer reveals where ordinal boundaries are fuzzy in real climate.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-034 — Input-gradient saliency (extended note)

Cheap first-order attribution highlights which scaled inputs move the selected logit most on a test subsample.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-035 — Optional Kernel SHAP in notebook (extended note)

Model-agnostic Shapley values provide complementary global importance rankings to saliency.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-036 — risk_lag_* excluded from forecast features (extended note)

Past risk labels are not known in true forecast mode; the production feature list avoids label leakage paths.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-037 — merge_auxiliary_features parity (extended note)

Evaluation merges the same optional humidity, NDVI, and merged-weather columns as the notebook when files exist.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-038 — PyTorch 2.x requirement (extended note)

Modern torch improves performance and API stability for checkpoint loading across CPU and CUDA.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-039 — CPU default in evaluation (extended note)

Reproducibility on laptops without GPUs; CUDA used automatically when available.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-040 — Ordinal cross-entropy versus CORAL (extended note)

CORAL or cumulative link models respect ordinality in the loss; cross-entropy with macro-F1 was chosen for simplicity and strong empirical fit.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-041 — Dropout placement (extended note)

Dropout after the RNN (where enabled by num_layers) and in the MLP head reduces overfitting to city-specific quirks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-042 — Hidden size sixty-four (extended note)

Balances capacity and data-limited risk; doubling did not justify extra parameters in exploratory runs documented in the notebook.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-043 — Two RNN layers (extended note)

Depth helps hierarchical temporal abstraction without making the stack as deep as ASR or language models.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-044 — Embedding dimension eight (extended note)

Small embeddings regularize city-specific capacity and avoid dominating the concatenated bidirectional context.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-045 — Batch size thirty-two (extended note)

Stable gradients for variable-length-like fixed windows grouped across cities; not so large as to under-utilize regularization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-046 — Eighty epoch cap with patience twelve (extended note)

Enough room for convergence while early stopping on plateauing validation macro-F1 prevents wasted compute.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-047 — Learning rate 3e-4 (extended note)

A conservative default for AdamW on scaled tabular sequence inputs; reduced on plateau.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-048 — Sequence construction per city (extended note)

Windows slide month by month; each sample uses twelve consecutive rows and predicts risk at the final month.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-049 — Unknown cities at inference (extended note)

Cities absent from training are dropped in evaluation to avoid undefined embedding indices.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-050 — FIG_DIR artifacts (extended note)

Confusion matrices and reports are written as PNG and TXT for versioned evidence in capstone and portfolio reviews.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-051 — FORECAST_DIR CSV contracts (extended note)

Horizon and scenario encoded in filenames so the Streamlit loader stays simple and cache-friendly.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-052 — Environmental variable SEQUENCE_CHECKPOINT_NAME (extended note)

Swap checkpoints without editing code for ablation studies or LSTM exports.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-053 — JSON metrics sidecar (extended note)

metrics.json documents pipeline mode and human notes beside binary pickles for reproducibility audits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-054 — Heat stress index z-scores (extended note)

Column-wise z-scoring within the frame at index construction stabilizes relative contributions before weighting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-055 — Risk label stability (extended note)

If HSI definition changes, recomputing quantiles is explicit; no hidden label drift inside the neural loss alone.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-056 — National versus city-specific urban metrics (extended note)

World Bank series lack intra-city variation but anchor cross-year trends used in scenario projections.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-057 — Humidity path optional (extended note)

When raw humidity exists, merges enrich the notebook model; base GRU feature set remains a strict core for train.py.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-058 — NDVI as vegetation stress proxy (extended note)

Optional monthly NDVI can indicate land-surface greenness changes correlated with heat and aridity patterns.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-059 — Merged scaled weather (extended note)

Optional processed panel can harmonize disparate instruments when available under data/processed.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-060 — forecast_lstm scenario knobs (extended note)

Population multiplier, urban percentage delta, and temperature delta propagate through recomputed HSI-like pathways.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-061 — Climatology projections in forecast (extended note)

Future months combine climatological baselines with anomaly medians and scenario deltas to synthesize plausible feature rows.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-062 — What-if probability rescaling in app (extended note)

Some UI explorations rescale saved probabilities for communication; they do not retrain the network online.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-063 — Ethical framing (extended note)

Model outputs are decision support, not clinical diagnoses or evacuation orders without local validation.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-064 — Data rights and provenance (extended note)

Users must respect upstream CSV licenses and refresh raw pulls for operational use.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-065 — Uncertainty quantification gap (extended note)

Softmax probabilities are not calibrated posterior beliefs unless post-hoc calibration is added.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-066 — Class boundary ambiguity (extended note)

Moderate versus High is inherently fuzzy when HSI sits near quantile cuts; expect adjacent-class errors.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-067 — Non-stationarity under climate change (extended note)

Train years may be cooler than future test decades; monitoring drift is essential.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-068 — Spatial autocorrelation (extended note)

Cities are not IID; national drivers induce correlation that inflates effective sample size less than row counts suggest.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-069 — Label noise from proxies (extended note)

Without mortality or morbidity linkage, Extreme risk is a statistical tail, not a verified health outcome.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-070 — Bidirectional caveat (extended note)

Strict real-time nowcasting from only past months might prefer unidirectional encoders; here the window is historical analysis.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-071 — Twelve-month minimum per sequence (extended note)

Cities shorter than the window after filtering contribute no sequences, trimming usable test counts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-072 — Evaluation uses merged features if present (extended note)

Column parity matters: missing columns are created then median-filled from train statistics.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-073 — Saliency subsampling (extended note)

Input-gradient importance uses a capped number of batches for speed on CPU evaluation runs.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-074 — Confusion matrix normalization (extended note)

Normalized matrices ease comparison across imbalanced classes versus raw counts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-075 — Extreme recall headline metric (extended note)

Policy users often prioritize catching dangerous tails even if overall accuracy dips slightly.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-076 — Weighted average versus macro in reports (extended note)

sklearn classification reports include both to show dominance of frequent classes versus balanced views.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-077 — Peak summer error analysis (extended note)

May–July buckets often concentrate heatwave signals; misclassifications there matter for early warnings.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-078 — Monsoon complexity (extended note)

Rainfall and cloud dynamics interact with temperature; residual errors may cluster in transitional months.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-079 — Winter low-risk dominance (extended note)

Low class precision and recall are high because many winter months are unambiguous.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-080 — City coordinate dictionary in app (extended note)

Static lat/lon anchors deck.gl maps for major cities in the dashboard layer.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-081 — PyDeck integration (extended note)

GPU-accelerated maps help communicate spatial risk narratives alongside time series.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-082 — Plotly for timelines (extended note)

Interactive hover supports exploratory analysis of scenario divergence across horizons.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-083 — Horizon grid six to seventy-two months (extended note)

Six-month steps balance file count with stakeholder talking points like five- or six-year outlooks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-084 — generate_forecasts entrypoint (extended note)

Wrapper scripts allow operators to choose lstm path consistently.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-085 — Capstone learning goals (extended note)

Demonstrates end-to-end ML engineering: data, modeling, evaluation, deployment, and honest limitations.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-086 — Repository layout heat-risk-pk (extended note)

Nested package keeps data and models colocated for coursework submission hygiene.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-087 — requirements.txt minimalism (extended note)

Only packages used across train, evaluate, notebook, and app are pinned with lower bounds.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-088 — No raw data in git LFS note (extended note)

Large raw CSVs may be excluded; document acquisition steps for reproducibility.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-089 — Notebook as experiment hub (extended note)

deep_learning_model_selection.ipynb records architecture zoo runs better than scattered scripts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-090 — data_processing.ipynb (extended note)

Optional EDA companion for coursework narrative.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-091 — joblib for feature list (extended note)

Simple pickle of column names aligns Python versions across sklearn-free inference paths.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-092 — pandas for I/O (extended note)

CSV-first pipeline keeps barrier to entry low for collaborators.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-093 — numpy float32 tensors (extended note)

Halves memory versus float64 for sequence batches without hurting classifier optimization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-094 — torch.nn.GRU bidirectional (extended note)

Doubles hidden dimension before attention; concatenated forward and backward states.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-095 — Attention softmax stability (extended note)

Softmax over sequence positions yields a proper convex combination of hidden states.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-096 — MLP head ReLU (extended note)

Nonlinearity before final logits enables curved decision boundaries in the concatenated embedding space.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-097 — Cross-entropy with integer labels (extended note)

Standard multiclass setup; ordinal structure captured indirectly via metric choice.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-098 — Stratification not used in RNN (extended note)

Sequences overlap; stratified batching is less straightforward than in IID tabular tasks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-099 — Shuffle training batches (extended note)

Random order of windows across cities reduces systematic gradient bias.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-100 — Weight decay on all AdamW parameters (extended note)

Including embeddings mildly regularizes city-specific overfitting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-101 — Random seed forty-two (extended note)

Common seed for numpy/torch where set improves repeatability across machines.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-102 — Logging patience in notebook (extended note)

Document early-stop epoch for transparency in capstone writeups.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-103 — Transformer positional encoding (extended note)

If used in notebook, sinusoidal or learned positions help but did not rescue the small-data regime here.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-104 — TCN dilation pattern (extended note)

Expanding receptive field captures multi-scale seasonality; kernel size and depth traded off in notebook.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-105 — Normalization layer choices (extended note)

Batch norm less common on tiny batch RNNs; dropout preferred in this stack.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-106 — Mixed precision not required (extended note)

Panel data batch cost is modest; full float32 simplifies numerics for coursework.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-107 — CUDA deterministic flags (extended note)

Not enforced; reproducibility relies primarily on seed and deterministic splits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-108 — Class counts reporting (extended note)

Always report support per class in confusion analysis to avoid misreading accuracy.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-109 — Risk communication color ramp (extended note)

Green to red ramp in UI encodes ordered severity intuitively.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-110 — Accessibility considerations (extended note)

Colorblind-safe palettes may be added; current ramp is simple defaults.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-111 — Time zone assumptions (extended note)

Monthly rows use calendar months without sub-monthly alignment across cities.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-112 — Leap years (extended note)

Daily aggregation handles variable month lengths via groupby; February lengths differ naturally.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-113 — Station relocation risk (extended note)

If a station moves, climatology nonstationarity may confuse anomaly features; metadata audit advised.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-114 — Urban heat island underrepresentation (extended note)

Single station per city may miss intra-urban gradients.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-115 — Population density units (extended note)

World Bank series documentation should be checked for units when interpreting magnitudes.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-116 — Urban percentage interpretation (extended note)

National urban share is a coarse proxy for city-specific sprawl and impervious cover.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-117 — Wind measurement height (extended note)

Anemometer standards vary; cross-city comparability is imperfect.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-118 — Precipitation in HSI (extended note)

Not directly in core HSI weights; rainfall may enter optional merges or future extensions.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-119 — Surface temperature data source (extended note)

Users should cite the exact surface temperature product in academic work.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-120 — Pakistan geographic focus (extended note)

Findings do not transfer automatically to other countries without retraining.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-121 — Climate adaptation planning (extended note)

Outputs can rank months for heat action budgeting when combined with local knowledge.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-122 — Public health integration pathway (extended note)

Hospital admissions or mortality could calibrate Extreme thresholds in future iterations.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-123 — Emergency management use (extended note)

Scenario tables can inform tabletop exercises even if not operational forecasts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-124 — Insurance and risk finance (extended note)

Tail probabilities might inform parametric triggers only after rigorous backtesting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-125 — Education and outreach (extended note)

Interactive app helps communicate compound urban heat risks to students.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-126 — Research extensions (extended note)

Graph neural networks across cities could exploit spatial edges explicitly.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-127 — Research extensions (extended note)

Hierarchical Bayesian models could quantify uncertainty more rigorously.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-128 — Research extensions (extended note)

Conformal prediction could yield distribution-free coverage statements for risk bins.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-129 — Operational monitoring (extended note)

Monthly refresh jobs could append rows and fine-tune with regularization to avoid catastrophic forgetting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-130 — Model registry pattern (extended note)

SEQUENCE_CHECKPOINT_NAME acts as a lightweight registry pointer for deployment.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-131 — Containerization future (extended note)

Dockerfile not included; add for production if required by IT policy.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-132 — CI testing future (extended note)

Smoke tests on evaluate_lstm with tiny synthetic CSV could guard regressions.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-133 — Versioned datasets (extended note)

Hash raw inputs in metrics.json for stronger provenance tracking.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-134 — Documentation maintenance (extended note)

Long READMEs should be regenerated or curated when hyperparameters change.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-135 — Temporal leakage and rolling windows (extended note)

Rolling means use a one-month shift so the current month's target does not leak smoothed information from the same calendar month in an improper way.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-136 — Year-based holdout versus random windows (extended note)

Random windows inside a city would let adjacent months appear in both train and test, inflating scores; chronological splits respect causality.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-137 — Macro-F1 versus accuracy for early stopping (extended note)

Accuracy is dominated by the majority class; macro-F1 forces the optimizer to care about rare Moderate/High/Extreme months.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-138 — Class-weighted cross-entropy (extended note)

Inverse-frequency or balanced weights reduce collapse to always predicting Low, which a naive baseline exhibits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-139 — Twelve-month context length (extended note)

Covers a full seasonal cycle while keeping the tensor small enough for limited data and stable optimization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-140 — Bidirectional RNN over the window (extended note)

The operational question is state at the end of a known segment; both directions summarize intra-window dynamics without using future labels.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-141 — Attention pooling over time (extended note)

Produces a weighted summary of which months in the year contributed most, aiding interpretability versus last-step pooling only.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-142 — City embeddings (extended note)

Cities differ in baseline climate and urban form; embeddings absorb residual city-specific effects not fully encoded in tabular columns.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-143 — GRU versus LSTM inductive bias (extended note)

GRU has fewer gates and often trains faster on modest panels; empirically it edged LSTM on validation macro-F1 here.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-144 — TCN as a convolutional alternative (extended note)

Parallel receptive fields over dilated convolutions stress a complementary inductive bias; results were competitive but slightly below GRU.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-145 — Transformer as a stress test (extended note)

Global mixing with limited data produced a negative result, illustrating that architecture hype must meet data scale and tuning.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-146 — AdamW and weight decay (extended note)

Decoupled weight decay regularizes large weights in the MLP head and embeddings without destabilizing momentum estimates.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-147 — ReduceLROnPlateau (extended note)

Lowering the learning rate when validation macro-F1 stalls helps late-stage fine-tuning without manual schedules.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-148 — Gradient clipping (extended note)

Recurrent stacks can explode gradients on heterogeneous city-month panels; clipping stabilizes training.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-149 — Median imputation fit on train only (extended note)

Using validation or test medians would leak information; train-only medians mirror deployment when new cities appear.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-150 — StandardScaler fit on train years only (extended note)

Same rationale as imputation: test statistics must never define the scaling used for test inputs.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-151 — Quantile thresholds for ordinal labels (extended note)

Global P50/P75/P90 splits on HSI yield four ordered tiers that are easy to audit and recompute if weights change.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-152 — HSI as expert-weighted composite (extended note)

Transparency for stakeholders beats opaque end-to-end label learning when health outcomes are not yet linked.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-153 — Wind enters with a negative coefficient (extended note)

Higher wind speeds modestly reduce composite stress in this simplified index; missing wind is median-filled before z-scoring.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-154 — Monthly aggregation from daily weather (extended note)

Means and sums summarize station behavior while controlling noise versus keeping every daily row.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-155 — Minimum months per city filter (extended note)

Cities with fewer than six hundred months are dropped to ensure long histories for sequence construction and robust climatology.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-156 — Minimum non-null tavg coverage (extended note)

Eighty percent coverage avoids cities where intermittent reporting would make z-scores and lags unreliable.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-157 — World Bank national drivers (extended note)

Population density and urban share are annual national proxies; coarse but consistent across cities for slow-changing exposure.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-158 — Surface temperature merge (extended note)

Regional-scale warming signal complements station means and anchors cross-city coherence in hot seasons.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-159 — Cyclical month encoding (extended note)

Sine and cosine preserve circular continuity of January and December unlike raw integer month.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-160 — Climatology and anomaly decomposition (extended note)

Separates expected seasonal warmth from deviations that often precede heat-stress escalations.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-161 — Recursive forecasting semantics (extended note)

Long horizons repeat seasonal structure under scenario knobs; interpret as stylized stress tests, not exact calendar futures.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-162 — Scenario temperature deltas (extended note)

Plus one and plus two degrees Celsius are simple, communicative warming knobs for comparative dashboards.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-163 — Streamlit for operations (extended note)

Rapid iteration for non-engineer stakeholders to explore precomputed forecasts without a heavy web stack.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-164 — Checkpoint payload structure (extended note)

Torch pickles bundle weights, config, feature list, and city index map so evaluation and forecast scripts stay in sync.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-165 — Majority baseline honesty (extended note)

Reporting a majority-class baseline shows class imbalance severity and justifies balanced metrics and weighted loss.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-166 — Per-city evaluation CSV (extended note)

Downstream UI and analysts can see heterogeneity: some cities have harder extreme-event detection than others.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-167 — Temporal error patterns by season bucket (extended note)

Aggregating errors by monsoon versus peak summer reveals where ordinal boundaries are fuzzy in real climate.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-168 — Input-gradient saliency (extended note)

Cheap first-order attribution highlights which scaled inputs move the selected logit most on a test subsample.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-169 — Optional Kernel SHAP in notebook (extended note)

Model-agnostic Shapley values provide complementary global importance rankings to saliency.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-170 — risk_lag_* excluded from forecast features (extended note)

Past risk labels are not known in true forecast mode; the production feature list avoids label leakage paths.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-171 — merge_auxiliary_features parity (extended note)

Evaluation merges the same optional humidity, NDVI, and merged-weather columns as the notebook when files exist.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-172 — PyTorch 2.x requirement (extended note)

Modern torch improves performance and API stability for checkpoint loading across CPU and CUDA.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-173 — CPU default in evaluation (extended note)

Reproducibility on laptops without GPUs; CUDA used automatically when available.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-174 — Ordinal cross-entropy versus CORAL (extended note)

CORAL or cumulative link models respect ordinality in the loss; cross-entropy with macro-F1 was chosen for simplicity and strong empirical fit.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-175 — Dropout placement (extended note)

Dropout after the RNN (where enabled by num_layers) and in the MLP head reduces overfitting to city-specific quirks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-176 — Hidden size sixty-four (extended note)

Balances capacity and data-limited risk; doubling did not justify extra parameters in exploratory runs documented in the notebook.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-177 — Two RNN layers (extended note)

Depth helps hierarchical temporal abstraction without making the stack as deep as ASR or language models.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-178 — Embedding dimension eight (extended note)

Small embeddings regularize city-specific capacity and avoid dominating the concatenated bidirectional context.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-179 — Batch size thirty-two (extended note)

Stable gradients for variable-length-like fixed windows grouped across cities; not so large as to under-utilize regularization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-180 — Eighty epoch cap with patience twelve (extended note)

Enough room for convergence while early stopping on plateauing validation macro-F1 prevents wasted compute.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-181 — Learning rate 3e-4 (extended note)

A conservative default for AdamW on scaled tabular sequence inputs; reduced on plateau.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-182 — Sequence construction per city (extended note)

Windows slide month by month; each sample uses twelve consecutive rows and predicts risk at the final month.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-183 — Unknown cities at inference (extended note)

Cities absent from training are dropped in evaluation to avoid undefined embedding indices.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-184 — FIG_DIR artifacts (extended note)

Confusion matrices and reports are written as PNG and TXT for versioned evidence in capstone and portfolio reviews.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-185 — FORECAST_DIR CSV contracts (extended note)

Horizon and scenario encoded in filenames so the Streamlit loader stays simple and cache-friendly.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-186 — Environmental variable SEQUENCE_CHECKPOINT_NAME (extended note)

Swap checkpoints without editing code for ablation studies or LSTM exports.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-187 — JSON metrics sidecar (extended note)

metrics.json documents pipeline mode and human notes beside binary pickles for reproducibility audits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-188 — Heat stress index z-scores (extended note)

Column-wise z-scoring within the frame at index construction stabilizes relative contributions before weighting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-189 — Risk label stability (extended note)

If HSI definition changes, recomputing quantiles is explicit; no hidden label drift inside the neural loss alone.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-190 — National versus city-specific urban metrics (extended note)

World Bank series lack intra-city variation but anchor cross-year trends used in scenario projections.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-191 — Humidity path optional (extended note)

When raw humidity exists, merges enrich the notebook model; base GRU feature set remains a strict core for train.py.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-192 — NDVI as vegetation stress proxy (extended note)

Optional monthly NDVI can indicate land-surface greenness changes correlated with heat and aridity patterns.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-193 — Merged scaled weather (extended note)

Optional processed panel can harmonize disparate instruments when available under data/processed.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-194 — forecast_lstm scenario knobs (extended note)

Population multiplier, urban percentage delta, and temperature delta propagate through recomputed HSI-like pathways.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-195 — Climatology projections in forecast (extended note)

Future months combine climatological baselines with anomaly medians and scenario deltas to synthesize plausible feature rows.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-196 — What-if probability rescaling in app (extended note)

Some UI explorations rescale saved probabilities for communication; they do not retrain the network online.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-197 — Ethical framing (extended note)

Model outputs are decision support, not clinical diagnoses or evacuation orders without local validation.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-198 — Data rights and provenance (extended note)

Users must respect upstream CSV licenses and refresh raw pulls for operational use.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-199 — Uncertainty quantification gap (extended note)

Softmax probabilities are not calibrated posterior beliefs unless post-hoc calibration is added.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-200 — Class boundary ambiguity (extended note)

Moderate versus High is inherently fuzzy when HSI sits near quantile cuts; expect adjacent-class errors.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-201 — Non-stationarity under climate change (extended note)

Train years may be cooler than future test decades; monitoring drift is essential.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-202 — Spatial autocorrelation (extended note)

Cities are not IID; national drivers induce correlation that inflates effective sample size less than row counts suggest.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-203 — Label noise from proxies (extended note)

Without mortality or morbidity linkage, Extreme risk is a statistical tail, not a verified health outcome.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-204 — Bidirectional caveat (extended note)

Strict real-time nowcasting from only past months might prefer unidirectional encoders; here the window is historical analysis.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-205 — Twelve-month minimum per sequence (extended note)

Cities shorter than the window after filtering contribute no sequences, trimming usable test counts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-206 — Evaluation uses merged features if present (extended note)

Column parity matters: missing columns are created then median-filled from train statistics.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-207 — Saliency subsampling (extended note)

Input-gradient importance uses a capped number of batches for speed on CPU evaluation runs.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-208 — Confusion matrix normalization (extended note)

Normalized matrices ease comparison across imbalanced classes versus raw counts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-209 — Extreme recall headline metric (extended note)

Policy users often prioritize catching dangerous tails even if overall accuracy dips slightly.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-210 — Weighted average versus macro in reports (extended note)

sklearn classification reports include both to show dominance of frequent classes versus balanced views.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-211 — Peak summer error analysis (extended note)

May–July buckets often concentrate heatwave signals; misclassifications there matter for early warnings.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-212 — Monsoon complexity (extended note)

Rainfall and cloud dynamics interact with temperature; residual errors may cluster in transitional months.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-213 — Winter low-risk dominance (extended note)

Low class precision and recall are high because many winter months are unambiguous.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-214 — City coordinate dictionary in app (extended note)

Static lat/lon anchors deck.gl maps for major cities in the dashboard layer.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-215 — PyDeck integration (extended note)

GPU-accelerated maps help communicate spatial risk narratives alongside time series.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-216 — Plotly for timelines (extended note)

Interactive hover supports exploratory analysis of scenario divergence across horizons.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-217 — Horizon grid six to seventy-two months (extended note)

Six-month steps balance file count with stakeholder talking points like five- or six-year outlooks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-218 — generate_forecasts entrypoint (extended note)

Wrapper scripts allow operators to choose lstm path consistently.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-219 — Capstone learning goals (extended note)

Demonstrates end-to-end ML engineering: data, modeling, evaluation, deployment, and honest limitations.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-220 — Repository layout heat-risk-pk (extended note)

Nested package keeps data and models colocated for coursework submission hygiene.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-221 — requirements.txt minimalism (extended note)

Only packages used across train, evaluate, notebook, and app are pinned with lower bounds.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-222 — No raw data in git LFS note (extended note)

Large raw CSVs may be excluded; document acquisition steps for reproducibility.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-223 — Notebook as experiment hub (extended note)

deep_learning_model_selection.ipynb records architecture zoo runs better than scattered scripts.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-224 — data_processing.ipynb (extended note)

Optional EDA companion for coursework narrative.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-225 — joblib for feature list (extended note)

Simple pickle of column names aligns Python versions across sklearn-free inference paths.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-226 — pandas for I/O (extended note)

CSV-first pipeline keeps barrier to entry low for collaborators.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-227 — numpy float32 tensors (extended note)

Halves memory versus float64 for sequence batches without hurting classifier optimization.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-228 — torch.nn.GRU bidirectional (extended note)

Doubles hidden dimension before attention; concatenated forward and backward states.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-229 — Attention softmax stability (extended note)

Softmax over sequence positions yields a proper convex combination of hidden states.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-230 — MLP head ReLU (extended note)

Nonlinearity before final logits enables curved decision boundaries in the concatenated embedding space.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-231 — Cross-entropy with integer labels (extended note)

Standard multiclass setup; ordinal structure captured indirectly via metric choice.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-232 — Stratification not used in RNN (extended note)

Sequences overlap; stratified batching is less straightforward than in IID tabular tasks.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-233 — Shuffle training batches (extended note)

Random order of windows across cities reduces systematic gradient bias.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-234 — Weight decay on all AdamW parameters (extended note)

Including embeddings mildly regularizes city-specific overfitting.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-235 — Random seed forty-two (extended note)

Common seed for numpy/torch where set improves repeatability across machines.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-236 — Logging patience in notebook (extended note)

Document early-stop epoch for transparency in capstone writeups.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-237 — Transformer positional encoding (extended note)

If used in notebook, sinusoidal or learned positions help but did not rescue the small-data regime here.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-238 — TCN dilation pattern (extended note)

Expanding receptive field captures multi-scale seasonality; kernel size and depth traded off in notebook.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-239 — Normalization layer choices (extended note)

Batch norm less common on tiny batch RNNs; dropout preferred in this stack.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-240 — Mixed precision not required (extended note)

Panel data batch cost is modest; full float32 simplifies numerics for coursework.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-241 — CUDA deterministic flags (extended note)

Not enforced; reproducibility relies primarily on seed and deterministic splits.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-242 — Class counts reporting (extended note)

Always report support per class in confusion analysis to avoid misreading accuracy.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-243 — Risk communication color ramp (extended note)

Green to red ramp in UI encodes ordered severity intuitively.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-244 — Accessibility considerations (extended note)

Colorblind-safe palettes may be added; current ramp is simple defaults.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-245 — Time zone assumptions (extended note)

Monthly rows use calendar months without sub-monthly alignment across cities.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-246 — Leap years (extended note)

Daily aggregation handles variable month lengths via groupby; February lengths differ naturally.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-247 — Station relocation risk (extended note)

If a station moves, climatology nonstationarity may confuse anomaly features; metadata audit advised.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-248 — Urban heat island underrepresentation (extended note)

Single station per city may miss intra-urban gradients.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-249 — Population density units (extended note)

World Bank series documentation should be checked for units when interpreting magnitudes.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-250 — Urban percentage interpretation (extended note)

National urban share is a coarse proxy for city-specific sprawl and impervious cover.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-251 — Wind measurement height (extended note)

Anemometer standards vary; cross-city comparability is imperfect.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-252 — Precipitation in HSI (extended note)

Not directly in core HSI weights; rainfall may enter optional merges or future extensions.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-253 — Surface temperature data source (extended note)

Users should cite the exact surface temperature product in academic work.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.

### DD-254 — Pakistan geographic focus (extended note)

Findings do not transfer automatically to other countries without retraining.

This decision is revisited when data volume grows, new auxiliary sensors arrive, or stakeholders require calibrated probabilities tied to health outcomes.



---

## Appendix B — Output artifacts reference

Common files under `heat-risk-pk/outputs/figures/` after evaluation:

- `model_metrics.csv` — baseline vs GRU metrics
- `best_model.txt` — top row by selection rule
- `classification_report_sequence.txt` — per-class precision/recall/F1/support
- `classification_report_baseline.txt` — baseline report
- `confusion_matrix_sequence.png` — normalized confusion matrix
- `confusion_matrix_baseline.png` — baseline confusion matrix
- `city_wise_accuracy.csv` — per-city diagnostics for UI tables
- `most_common_misclassifications.csv` — top error modes
- `extreme_error_summary.csv` — binary extreme detection summary
- `class_specific_performance.csv` — class-level cards
- `temporal_error_patterns.csv` — season-bucket diagnostics
- `sequence_feature_saliency.csv` / `sequence_feature_saliency_top15.png` — saliency exports

Forecasts under `heat-risk-pk/outputs/forecasts/` follow `forecast_{H}m_{scenario}.csv` naming.

---

## Appendix C — Conceptual viva preparation

**Likely question:** “Why not predict temperature directly?”  
**Answer:** The project targets **risk tiers** aligned to stakeholder communication; temperature alone misses exposure and slow drivers unless expanded.

**Likely question:** “Is bidirectional cheating?”  
**Answer:** No future **labels** are used; the window is fully observed history ending at prediction month, matching the operational encoding task.

**Likely question:** “What is your biggest limitation?”  
**Answer:** Labels are not outcome-linked; scenario forecasts are stylized; spatial intra-city variability is not captured.

---


Developed as **CS-419 Capstone** work: **Urban Heat Stress Risk Forecasting — Pakistan**. 

---

*End of README.*


