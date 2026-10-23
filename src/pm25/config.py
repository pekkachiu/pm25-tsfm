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

# (latitude, longitude) from MOENV aqx_p_07 station metadata (TWD97 = WGS84 to ~1 m).
STATION_COORDS = {
    "古亭": (25.020608, 121.529556), "板橋": (25.012972, 121.458667),
    "桃園": (24.994711, 121.305005), "忠明": (24.151958, 120.641092),
    "西屯": (24.162197, 120.616917), "彰化": (24.066000, 120.541519),
    "斗六": (23.711853, 120.544994), "嘉義": (23.464779, 120.441251),
    "臺南": (22.989283, 120.219479), "前金": (22.633903, 120.286761),
    "小港": (22.565833, 120.337736), "屏東": (22.673081, 120.488033),
}

# English site names (aqx_p_07 siteengname) for figures and the English report.
STATION_EN = {
    "古亭": "Guting", "板橋": "Banqiao", "桃園": "Taoyuan", "忠明": "Zhongming",
    "西屯": "Xitun", "彰化": "Changhua", "斗六": "Douliu", "嘉義": "Chiayi",
    "臺南": "Tainan", "前金": "Qianjin", "小港": "Xiaogang", "屏東": "Pingtung",
}

# Realistic future wind: Open-Meteo archived NWP runs, value issued 24 h before valid time
# (previous_day1), so every forecast for origin+1..+24 h was published by the origin.
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WIND_FORECAST_DIR = "data/wind_forecast"  # one sub-directory per NWP model
# NWP model -> first date to download. JMA GSM is archived back to 2018, so trained baselines
# can learn from forecasts over 2021-2024; best_match ("seamless") starts in 2024 and only
# covers the 512 h context of the first 2025 origin.
NWP_MODELS = {"jma_gsm": "2020-12-01", "best_match": "2024-12-01"}
PRIMARY_NWP = "jma_gsm"
WIND_FORECAST_COVARIATES = ("fc_wind_speed", "fc_wind_sin", "fc_wind_cos")

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

# Lower bound of AQI 101-150 for PM2.5 in MOENV's AQI table effective 2025-01-01
# (airtw.moenv.gov.tw AirQualityIndicatorNew); applied here to hourly concentrations.
PM25_AQI100_THRESHOLD = 30.5

# Northeast-monsoon season (wind-driven episodes, E2) vs the rest of the year.
AUTUMN_WINTER_MONTHS = (10, 11, 12, 1, 2, 3)

ADWIN_DELTA = 0.002  # river's default confidence for ADWIN

# Phase 3: fine-tuning one global Chronos-2. Train 2021-2023, select the learning rate and
# checkpoint on 2024, test on 2025 like every other model.
FINETUNE_TRAIN = ("2021-01-01", "2023-12-31 23:00")
FINETUNE_VALID = ("2024-01-01", "2024-12-31 23:00")
FINETUNE_LEARNING_RATES = (1e-6, 1e-5)  # full fine-tuning; 1e-6 is the Chronos-2 default
FINETUNE_STEPS = 1000  # Chronos2Pipeline.fit default; best checkpoint by 2024 loss
FINETUNE_BATCH_SIZE = 256
FINETUNE_VAL_EVERY_HOURS = 72  # validation windows every 3 days of 2024, per station
FINETUNED_DIR = "results/finetuned"

# Adaptive conformal inference: margin step in ug/m3 per observed outcome, fixed a priori
# (not tuned on the test year). See pm25.evaluation.conformal.
ACI_STEP = 0.05

# Diebold-Mariano tests on absolute-error differentials (model, baseline). Hourly origins
# overlap and PM2.5 errors persist for days, so the Newey-West lag is two days.
DM_HAC_LAG = 48
DM_PAIRS = (
    ("xgboost", "persistence"),
    ("lstm", "xgboost"),
    ("chronos_bolt_base", "chronos_bolt_small"),
    ("chronos2_uni", "persistence"),
    ("chronos2_uni", "xgboost"),
    ("chronos2_cov", "xgboost"),
    ("chronos2_cov", "chronos2_uni"),
    # Phase 1: gain from day-ahead NWP wind within each model, then NWP-vs-NWP.
    ("xgboost_nwp", "xgboost"),
    ("lstm_nwp", "lstm"),
    ("chronos2_wind_nwp", "chronos2_cov"),
    ("chronos2_wind_nwp", "xgboost_nwp"),
    ("chronos2_wind_bestmatch", "chronos2_wind_nwp"),
    ("chronos2_wind_oracle", "chronos2_wind_nwp"),
    ("chronos2_wind_oracle", "xgboost"),
)
