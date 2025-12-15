# Urban Heat Stress Risk Forecasting for Pakistani Cities

## Overview
This project forecasts **urban heat stress risk** for major Pakistani cities and provides scenario-based forecasts for the next **6/12/24 months**. It integrates multi-source climate and urbanization data and outputs a deployable PoC (Streamlit dashboard).

## Stakeholders
- Disaster preparedness agencies (NDMA/PDMA)
- City administrations and planners
- Public health departments
- NGOs and relief organizations

## Datasets (4 sources)
1. **Pakistan city daily weather** (city-level daily observations)  
2. **World Bank population density** (Pakistan yearly)  
3. **World Bank urban population %** (Pakistan yearly)  
4. **Average monthly surface temperature** (Pakistan monthly)

> All sources are public; see report references for citations.

## Repository Structure
- `src/` data pipeline + training + forecasting + explainability
- `data/raw/` place raw datasets here (not committed if large)
- `data/processed/` generated modeling datasets
- `models/` trained models + feature column lists
- `outputs/forecasts/` forecast CSVs
- `outputs/figures/` plots for report

## Setup
```bash
pip install -r requirements.txt
# Urban Heat Stress Risk Forecasting for Pakistani Cities

## Overview
This project forecasts **urban heat stress risk** for major Pakistani cities and provides scenario-based forecasts for the next **6/12/24 months**. It integrates multi-source climate and urbanization data and outputs a deployable PoC (Streamlit dashboard).

## Stakeholders
- Disaster preparedness agencies (NDMA/PDMA)
- City administrations and planners
- Public health departments
- NGOs and relief organizations

## Datasets (4 sources)
1. **Pakistan city daily weather** (city-level daily observations)  
2. **World Bank population density** (Pakistan yearly)  
3. **World Bank urban population %** (Pakistan yearly)  
4. **Average monthly surface temperature** (Pakistan monthly)

> All sources are public; see report references for citations.

## Repository Structure
- `src/` data pipeline + training + forecasting + explainability
- `data/raw/` place raw datasets here (not committed if large)
- `data/processed/` generated modeling datasets
- `models/` trained models + feature column lists
- `outputs/forecasts/` forecast CSVs
- `outputs/figures/` plots for report

## Setup
```bash
pip install -r requirements.txt

# Urban Heat Stress Risk Forecasting for Pakistani Cities

## Overview
This project forecasts **urban heat stress risk** for major Pakistani cities and provides scenario-based forecasts for the next **6/12/24 months**. It integrates multi-source climate and urbanization data and outputs a deployable PoC (Streamlit dashboard).

## Stakeholders
- Disaster preparedness agencies (NDMA/PDMA)
- City administrations and planners
- Public health departments
- NGOs and relief organizations

## Datasets (4 sources)
1. **Pakistan city daily weather** (city-level daily observations)  
2. **World Bank population density** (Pakistan yearly)  
3. **World Bank urban population %** (Pakistan yearly)  
4. **Average monthly surface temperature** (Pakistan monthly)

> All sources are public; see report references for citations.

## Repository Structure
- `src/` data pipeline + training + forecasting + explainability
- `data/raw/` place raw datasets here (not committed if large)
- `data/processed/` generated modeling datasets
- `models/` trained models + feature column lists
- `outputs/forecasts/` forecast CSVs
- `outputs/figures/` plots for report

## Setup
```bash
pip install -r requirements.txt

Train models and build processed datasets:
python -m src.train
Evaluate and generate plots:
python -m src.evaluate
python -m src.explain

How to Run (End-to-End)
Put raw files in data/raw/:
pakistan_city_weather_daily.csv
API_EN.POP.DNST_DS2_en_csv_v2_110190.csv
API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_110318.csv
average-monthly-surface-temperature.csv
Train models and build processed datasets:
python -m src.train
Evaluate and generate plots:
python -m src.evaluate
python -m src.explain
Generate forecasts (baseline + +1°C + +2°C):
python -m src.forecast
Models
We train two model families:
Monitoring model (Logistic Regression): uses lagged risk features for early warning monitoring.
Forecast model (Gradient Boosting): climate-only, suitable for forward forecasting and scenario analysis.

python -m src.train
Evaluate and generate plots:
python -m src.evaluate
python -m src.explain
Generate forecasts (baseline + +1°C + +2°C):
python -m src.forecast
Models
We train two model families:
Monitoring model (Logistic Regression): uses lagged risk features for early warning monitoring.
Forecast model (Gradient Boosting): climate-only, suitable for forward forecasting and scenario analysis.

```python
!python -m heat-risk-pk.src.train
!python -m heat-risk-pk.src.evaluate
!python -m heat-risk-pk.src.explain
!python -m heat-risk-pk.src.forecast
(Or set your working directory to heat-risk-pk and run python -m src.train etc.)