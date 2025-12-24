#  Urban Heat Stress Risk Forecasting — Pakistan (CS-245 Capstone)

End-to-end ML decision-support system to **forecast urban heat stress risk** for major Pakistani cities using **multi-source heterogeneous datasets** and a deployed **Streamlit PoC dashboard**.

##  What this project delivers
- **Integrated dataset pipeline** (4 sources merged + feature engineering)
- **ML pipeline** with baseline + multiple classical models + ensemble
- **Model comparison** (metrics + confusion matrices + classification reports)
# Urban Heat Stress Risk Forecasting — Pakistan (CS-245 Capstone)

Comprehensive end-to-end project that builds a decision-support system to forecast monthly urban heat stress risk for major Pakistani cities. It includes data ingestion, preprocessing, feature engineering, training, evaluation, explainability, forecast generation for scenarios, and a Streamlit proof-of-concept dashboard.

Repository root highlights

- `heat-risk-pk/` : main project folder
  - `app/app.py` : Streamlit dashboard
  - `data/raw/` : original source files (weather, World Bank, surface temp)
  - `data/processed/` : processed datasets used for modeling
  - `models/` : trained model artifacts and feature column lists
  - `outputs/forecasts/` : generated forecast CSVs (used by dashboard)
  - `outputs/figures/` : evaluation, SHAP and importance plots
  - `src/` : processing, modeling, evaluation, forecasting scripts
  - `requirements.txt` : Python dependencies

---

## Quick summary — what the project does

- Builds a monthly city-level dataset from daily weather and national indicators
- Computes a Heat Stress Index (HSI) and categorizes risk into 4 classes (Low/Moderate/High/Extreme)
- Trains models for two operational tasks:
  - Forecasting (climate-only HistGradientBoosting) — produces multi-month forecasts and scenarios
  - Monitoring (Logistic Regression) — short-term alerts using recent observed labels (risk_lag_*)
- Produces explainability artifacts (SHAP, permutation importance)
- Exposes results in an interactive Streamlit dashboard with maps, timelines, scenario comparison and what-if sliders

---

## Data sources (where to look)

- City daily weather: `heat-risk-pk/data/raw/pakistan_city_weather_daily.csv`
- World Bank population density: `heat-risk-pk/data/raw/API_EN.POP.DNST_DS2_en_csv_v2_110190.csv`
- World Bank urban population %: `heat-risk-pk/data/raw/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_110318.csv`
- Average monthly surface temperature: `heat-risk-pk/data/raw/average-monthly-surface-temperature.csv`

Loader functions are in `src/io.py`.

---

## How the pipeline works (high level)

1. Raw daily weather is aggregated to monthly city-level features (`src/preprocess.py`).
2. World Bank and surface temperature sources are converted to year/month form and merged.
3. Feature engineering adds:
   - Seasonal encodings (`month_sin`, `month_cos`)
   - City climatology and anomalies (`tavg_clim`, `tavg_anom`)
   - Z-scored inputs used by the Heat Stress Index (HSI)
4. Targets are created:
   - HSI (weighted z-score combination) in `src/targets.py`
   - Risk label via percentile bins (P50, P75, P90)
5. Lag and rolling features (`heat_lag_*`, `risk_lag_*`, rolling means/stds) are produced in `src/split.py`.
6. Two datasets are prepared:
   - Climate-only forecast dataset: `data/processed/df_model_forecast.csv` (no `risk_lag_*`)
   - Monitoring dataset: `data/processed/df_model_monitoring.csv` (includes `risk_lag_*`)

---

## Models and their roles

- Forecast model (deployed for dashboard): `models/forecast_hgb.pkl`
  - HistGradientBoosting classifier trained on climate-only features (no `risk_lag_*`).
  - Used for multi-month projections and scenario comparisons.

- Monitoring model: `models/monitoring_logreg.pkl`
  - Logistic Regression pipeline (scaler + LR) trained on dataset that includes recent observed `risk_lag_*`.
  - Intended for short-term monitoring/alerts where recent labels exist.

- Explainability helper: `models/explain_rf.pkl` (Random Forest) used to compute SHAP summaries.

Model definitions and pipelines live in `src/model_zoo.py` and training/evaluation in `src/train.py` and `src/evaluate.py`.

---

## Evaluation (key metrics)

- Evaluation artifacts and confusion matrices are saved to `outputs/figures/`.
- Example metrics (found at `outputs/figures/model_metrics.csv`):
  - HGB (forecast): macro-F1 ≈ 0.90, accuracy ≈ 0.92
  - Logistic Regression (monitoring): macro-F1 ≈ 0.91, accuracy ≈ 0.92 (benefits from `risk_lag_*`)

Notes: LR performs well on monitoring because it leverages recent labels (`risk_lag_*`) — this is valid for alerting but would be label leakage for forward forecasting. HGB is the correct choice for forecasting because it does not require future labels.

---

## Reproducible run order (recommended)

1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # mac / linux
# .venv\Scripts\activate   # windows
pip install -r heat-risk-pk/requirements.txt
```

2) Build processed datasets and model-ready tables

```bash
python heat-risk-pk/src/train.py --build-only
```

Note: `src/train.py` includes `build_dataset()` which loads raw data via `src/io.py`, aggregates and merges sources, applies feature engineering and creates `data/processed/df_model_forecast.csv` and `df_model_monitoring.csv`.

3) Train models and save artifacts

```bash
python heat-risk-pk/src/train.py
```

This trains monitoring and forecast models, saves `models/monitoring_logreg.pkl`, `models/forecast_hgb.pkl`, and writes `models/feature_cols_*.pkl`. It also saves processed CSVs to `data/processed/`.

4) Evaluate & produce figures (confusion matrices, metrics)

```bash
python heat-risk-pk/src/evaluate.py
```

5) Generate forecasts (scenarios & horizons)

```bash
python heat-risk-pk/src/forecast.py
```

This creates forecast CSVs in `outputs/forecasts/` used by the Streamlit dashboard.

6) (Optional) Explainability & SHAP plots

```bash
python heat-risk-pk/src/explain.py
```

7) Run the dashboard

```bash
streamlit run heat-risk-pk/app/app.py
```

Open the displayed URL in your browser (usually `http://localhost:8501`).

---

## Dashboard notes

- The dashboard reads forecast CSVs from `outputs/forecasts/` and figures from `outputs/figures/`.
- It shows city maps, timelines, scenario comparisons (+1°C, +2°C), and live what-if adjustments (re-scoring; not retraining).
- Long-horizon forecasts (≥12 months) are generated recursively and will stabilize into repeating seasonal cycles — see `src/forecast.py` and `Explanations/WHY_FORECASTS_REPEAT.md` for details.

---

## Troubleshooting

- Missing forecast files: run `python heat-risk-pk/src/forecast.py` to regenerate and verify `outputs/forecasts/`.
- Missing trained model `.pkl`: run `python heat-risk-pk/src/train.py`.
- Data memory/performance: consider running preprocessing on a machine with sufficient RAM; reduce sample size for a quick demo.

---

## Limitations & next steps

- Long-horizon forecasts stabilize into seasonal cycles due to recursive lags and static climatology.
- Surface temperature is national-level; city-scale climate projections would improve realism.
- Future work: integrate CMIP6 projections, add explicit year/time-trend features, test LightGBM/XGBoost ensembles, calibrate classes against health outcomes.

---

## Author

- Muhammad Abdullah Waqar


---

For a detailed project report (with formulas, code references, and appendix), see `heat-risk-pk/ML_Project_Final_Report.pdf`


