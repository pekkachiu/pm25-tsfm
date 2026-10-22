import numpy as np
import pandas as pd
import pytest


def daily_record(station, item, date, values):
    """Mimics one aqx_p_13 record: one item, one day, 24 hourly columns (strings)."""
    hours = {f"monitorvalue{h:02d}": v for h, v in enumerate(values)}
    return {"siteid": "14", "sitename": station, "itemengname": item,
            "monitordate": date, **hours}


@pytest.fixture
def hourly_index():
    return pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="Asia/Taipei")


@pytest.fixture
def raw_records():
    def day(item, first, date="2024-01-01"):
        return daily_record("古亭", item, date, [first, *[str(h) for h in range(1, 24)]])
    return [
        day("PM2.5", "x"), day("PM10", "20"), day("NO2", "-"), day("O3", "30"),
        day("WIND_SPEED", "1.5"), day("WIND_DIREC", "90"),
        day("AMB_TEMP", "25"),  # items outside the experiment are ignored
        daily_record("板橋", "PM2.5", "2024-01-01 ", ["-3", *["5"] * 23]),
    ]


@pytest.fixture
def rng():
    return np.random.default_rng(0)
