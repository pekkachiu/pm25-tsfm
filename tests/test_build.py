import numpy as np
import pandas as pd
import pytest

from pm25.data.build import build_station, missing_report


def _write_raw(raw_dir, station, year, hours, pm25=None):
    times = pd.date_range(f"{year}-01-01 01:00", periods=hours, freq="h")
    values = pm25 if pm25 is not None else [str(i) for i in range(hours)]
    pd.DataFrame({
        "sitename": station, "county": "x", "pm2.5": values, "pm10": "1", "no2": "1",
        "o3": "1", "windspeed": "1", "winddirec": "90",
        "datacreationdate": times.strftime("%Y-%m-%d %H:%M"),
    }).to_csv(raw_dir / f"{station}_{year}.csv", index=False)


def test_build_station_concatenates_years(tmp_path):
    _write_raw(tmp_path, "古亭", 2021, 24)
    _write_raw(tmp_path, "古亭", 2022, 24)
    df = build_station(tmp_path, "古亭", years=(2021, 2022))
    assert df.index.min() == pd.Timestamp("2021-01-01 01:00", tz="Asia/Taipei")
    assert df.index.max() == pd.Timestamp("2022-01-02 00:00", tz="Asia/Taipei")
    # one-year gap between the files stays missing
    assert df["pm25"].isna().sum() > 0
    assert {"pm25", "wind_sin", "wind_cos", "station"} <= set(df.columns)


def test_build_station_missing_file_raises(tmp_path):
    _write_raw(tmp_path, "古亭", 2021, 24)
    with pytest.raises(FileNotFoundError, match="古亭_2022"):
        build_station(tmp_path, "古亭", years=(2021, 2022))


def test_missing_report(tmp_path):
    values = ["1", "x", "3", "x", "x", "x", "x", "8"]
    _write_raw(tmp_path, "古亭", 2021, 8, pm25=values)
    df = build_station(tmp_path, "古亭", years=(2021,))
    report = missing_report(df)
    row = report.loc[2021]
    assert row["hours"] == 8
    assert row["pm25_missing"] == pytest.approx(4 / 8)  # 1 short gap filled, 4-gap stays
