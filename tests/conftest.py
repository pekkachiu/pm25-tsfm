import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_index():
    return pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="Asia/Taipei")


@pytest.fixture
def raw_records():
    """Mimics records returned by the MOENV AQX_P_488 API (all values are strings)."""
    return [
        {"sitename": "古亭", "county": "臺北市", "pm2.5": "12", "pm10": "20", "no2": "15",
         "o3": "30", "wind_speed": "1.5", "wind_direc": "90", "datacreationdate": "2024-01-01 01:00"},
        {"sitename": "古亭", "county": "臺北市", "pm2.5": "x", "pm10": "21", "no2": "-",
         "o3": "31", "wind_speed": "1.6", "wind_direc": "0", "datacreationdate": "2024-01-01 00:00"},
        {"sitename": "板橋", "county": "新北市", "pm2.5": "-3", "pm10": "", "no2": "16",
         "o3": "NR", "wind_speed": "2.0", "wind_direc": "180", "datacreationdate": "2024-01-01 00:00"},
    ]


@pytest.fixture
def rng():
    return np.random.default_rng(0)
