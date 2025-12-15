from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_DIR = PROJECT_ROOT / "outputs" / "forecasts"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

RANDOM_SEED = 42

# Files (raw)
WEATHER_FILE = "pakistan_city_weather_daily.csv"
POP_DENSITY_FILE = "API_EN.POP.DNST_DS2_en_csv_v2_110190.csv"
URBAN_PCT_FILE = "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_110318.csv"
SURFACE_TEMP_FILE = "average-monthly-surface-temperature.csv"

# Cities selection thresholds
MIN_MONTHS_PER_CITY = 600
MIN_TAVG_NONNULL = 0.80

# Temporal split
TRAIN_END_YEAR = 2015
VAL_END_YEAR = 2019

# Risk label percentiles
P50 = 0.50
P75 = 0.75
P90 = 0.90
