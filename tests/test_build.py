import numpy as np
import pandas as pd
import pytest

from conftest import daily_record
from pm25.data.build import build_station, missing_report


def _write_raw(raw_dir, station, year, pm25=None, day="01-01"):
    """One day of aqx_p_13 records for every experiment item."""
    pm25 = pm25 if pm25 is not None else [str(h) for h in range(24)]
    other = ["1"] * 24
    records = [daily_record(station, item, f"{year}-{day}", pm25 if item == "PM2.5" else other)
               for item in ("PM2.5", "PM10", "NO2", "O3", "WIND_SPEED", "WIND_DIREC")]
    pd.DataFrame.from_records(records).to_csv(raw_dir / f"{station}_{year}.csv", index=False)


def test_build_station_concatenates_years(tmp_path):
    _write_raw(tmp_path, "古亭", 2021)
    _write_raw(tmp_path, "古亭", 2022)
    df = build_station(tmp_path, "古亭", years=(2021, 2022))
    assert df.index.min() == pd.Timestamp("2021-01-01 01:00", tz="Asia/Taipei")
    assert df.index.max() == pd.Timestamp("2022-01-02 00:00", tz="Asia/Taipei")

    # one-year gap between the files stays missing
    assert df["pm25"].isna().sum() > 0
    assert {"pm25", "wind_sin", "wind_cos", "station"} <= set(df.columns)


def test_build_station_trims_to_requested_years(tmp_path):
    _write_raw(tmp_path, "古亭", 2021, day="12-31")
    df = build_station(tmp_path, "古亭", years=(2021,))
    # hour ending 2022-01-01 00:00 belongs to the 2021-12-31 record but outside 2021
    assert df.index.max().year == 2021


def test_build_station_missing_file_raises(tmp_path):
    _write_raw(tmp_path, "古亭", 2021)
    with pytest.raises(FileNotFoundError, match="古亭_2022"):
        build_station(tmp_path, "古亭", years=(2021, 2022))


def test_missing_report(tmp_path):
    values = ["1", "x", "3", "x", "x", "x", "x", *["8"] * 17]
    _write_raw(tmp_path, "古亭", 2021, pm25=values)
    df = build_station(tmp_path, "古亭", years=(2021,))
    report = missing_report(df)
    row = report.loc[2021]
    assert row["hours"] == 24
    assert row["pm25_missing"] == pytest.approx(4 / 24)  # 1 short gap filled, 4-gap stays  # 1 short gap filled, 4-gap stays
