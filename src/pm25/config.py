"""Single source of truth for experiment constants (see HW1_187 proposal)."""

TIMEZONE = "Asia/Taipei"

MOENV_API_BASE = "https://data.moenv.gov.tw/api/v2"
MOENV_DATASET = "aqx_p_488"
MOENV_API_KEY_ENV = "MOENV_API_KEY"

# Station selection: 3 stations per air-quality region named in the proposal.
STATIONS = {
    "northern": ("古亭", "板橋", "桃園"),
    "central": ("忠明", "西屯", "彰化"),
    "yunchianan": ("斗六", "嘉義", "臺南"),
    "kaoping": ("前金", "小港", "屏東"),
}

# Raw API field -> canonical column name.
RAW_COLUMNS = {
    "sitename": "station",
    "datacreationdate": "time",
    "pm2.5": "pm25",
    "pm10": "pm10",
    "no2": "no2",
    "o3": "o3",
    "windspeed": "wind_speed",
    "winddirec": "wind_dir",
}
TARGET = "pm25"
MEASUREMENTS = ("pm25", "pm10", "no2", "o3", "wind_speed", "wind_dir")
COVARIATES = ("pm10", "no2", "o3", "wind_speed", "wind_sin", "wind_cos")
WIND_COVARIATES = ("wind_speed", "wind_sin", "wind_cos")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

MAX_INTERP_GAP_HOURS = 3

YEARS = (2021, 2022, 2023, 2024, 2025)
TEST_YEAR = 2025

HORIZONS = (1, 6, 24)
CONTEXT_LENGTH = 512
ORIGIN_HOURS = (0,)
QUANTILE_LEVELS = (0.1, 0.5, 0.9)  # 80% prediction interval + median

# PM2.5 concentration (ug/m3) at which Taiwan's AQI exceeds 100.
# TODO: verify against the current MOENV AQI breakpoint table before final runs.
PM25_AQI100_THRESHOLD = 30.5
