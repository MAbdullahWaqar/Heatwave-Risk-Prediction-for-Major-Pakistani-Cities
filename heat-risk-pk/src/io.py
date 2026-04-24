import pandas as pd
from .config import (
    DATA_RAW,
    WEATHER_FILE,
    HUMIDITY_FILE,
    NDVI_FILE,
    POP_DENSITY_FILE,
    URBAN_PCT_FILE,
    SURFACE_TEMP_FILE,
)

def load_weather():
    return pd.read_csv(DATA_RAW / WEATHER_FILE)


def load_humidity():
    return pd.read_csv(DATA_RAW / HUMIDITY_FILE)


def load_ndvi():
    return pd.read_csv(DATA_RAW / NDVI_FILE)

def load_worldbank_pop_density():
    return pd.read_csv(DATA_RAW / POP_DENSITY_FILE, skiprows=4)

def load_worldbank_urban_pct():
    return pd.read_csv(DATA_RAW / URBAN_PCT_FILE, skiprows=4)

def load_surface_temp():
    return pd.read_csv(DATA_RAW / SURFACE_TEMP_FILE)
