"""Single source of truth for experiment constants (see HW1_187 proposal)."""

TIMEZONE = "Asia/Taipei"

MOENV_API_BASE = "https://data.moenv.gov.tw/api/v2"
# aqx_p_13: hourly station values, one record per (station, item, day), back to 2013.
# aqx_p_488 lacks hourly PM10/O3 before 2025-04, so it is not used.
MOENV_DATASET = "aqx_p_13"
MOENV_API_KEY_ENV = "MOENV_API_KEY"

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
PREDICTIONS_DIR = "results/predictions"
TABLES_DIR = "results/tables"
FIGURES_DIR = "results/figures"

# Station selection: 3 stations per air-quality region named in the proposal.
STATIONS = {
    "northern": ("古亭", "板橋", "桃園"),
    "central": ("忠明", "西屯", "彰化"),
    "yunchianan": ("斗六", "嘉義", "臺南"),
    "kaoping": ("前金", "小港", "屏東"),
}

# aqx_p_13 item code -> canonical column name.
ITEMS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "O3": "o3",
    "WIND_SPEED": "wind_speed",
    "WIND_DIREC": "wind_dir",
}
TARGET = "pm25"
MEASUREMENTS = tuple(ITEMS.values())
COVARIATES = ("pm10", "no2", "o3", "wind_speed", "wind_sin", "wind_cos")
WIND_COVARIATES = ("wind_speed", "wind_sin", "wind_cos")

MAX_INTERP_GAP_HOURS = 3

YEARS = (2021, 2022, 2023, 2024, 2025)
TEST_YEAR = 2025

HORIZONS = (1, 6, 24)
FORECAST_LENGTH = max(HORIZONS)  # every origin forecasts 24 steps; HORIZONS are lead times
CONTEXT_LENGTH = 512
ORIGIN_HOURS = tuple(range(24))  # a rolling origin every hour
# Windows need a complete target and complete last 48h of context (XGBoost lag range);
# older context gaps stay NaN (Chronos handles them natively).
REQUIRED_RECENT_HOURS = 48
QUANTILE_LEVELS = (0.1, 0.5, 0.9)  # 80% prediction interval + median

# PM2.5 concentration (ug/m3) at which Taiwan's AQI exceeds 100.
# TODO: verify against the current MOENV AQI breakpoint table before final runs.
PM25_AQI100_THRESHOLD = 30.5

# Northeast-monsoon season (wind-driven episodes, E2) vs the rest of the year.
AUTUMN_WINTER_MONTHS = (10, 11, 12, 1, 2, 3)

ADWIN_DELTA = 0.002  # river's default confidence for ADWIN
