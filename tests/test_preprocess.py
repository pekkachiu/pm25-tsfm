import numpy as np
import pandas as pd
import pytest

from conftest import daily_record
from pm25.data.preprocess import (
    encode_wind_direction,
    fill_short_gaps,
    preprocess_station,
    records_to_frame,
)


class TestRecordsToFrame:
    def test_hour_columns_are_hour_ending(self, raw_records):
        # monitorvalueHH on day D is the hour ending at D (HH+1):00 (matches aqx_p_488).
        df = records_to_frame(raw_records)
        guting = df[df["station"] == "古亭"]
        assert guting["time"].min() == pd.Timestamp("2024-01-01 01:00", tz="Asia/Taipei")
        assert guting["time"].max() == pd.Timestamp("2024-01-02 00:00", tz="Asia/Taipei")
        assert len(guting) == 24

    def test_pivots_items_to_canonical_columns(self, raw_records):
        df = records_to_frame(raw_records)
        for col in ["station", "time", "pm25", "pm10", "no2", "o3", "wind_speed", "wind_dir"]:
            assert col in df.columns
        assert "AMB_TEMP" not in df.columns
        row = df[(df["station"] == "古亭") & (df["time"].dt.hour == 2)].iloc[0]
        assert row["pm25"] == 1.0 and row["wind_speed"] == 1.0

    def test_invalid_flags_and_negatives_become_nan(self, raw_records):
        df = records_to_frame(raw_records)
        first = df[df["station"] == "古亭"].iloc[0]
        assert np.isnan(first["pm25"]) and np.isnan(first["no2"])
        assert np.isnan(df[df["station"] == "板橋"].iloc[0]["pm25"])

    def test_missing_items_become_nan_columns(self, raw_records):
        df = records_to_frame([r for r in raw_records if r["sitename"] == "板橋"])
        assert df["o3"].isna().all()

    def test_duplicate_records_merge_first_non_null(self, raw_records):
        empty = daily_record("古亭", "PM10", "2024-01-01", [None] * 24)
        df = records_to_frame([empty, *raw_records])
        assert df[df["station"] == "古亭"]["pm10"].iloc[0] == 20.0

    def test_time_is_tz_aware_taipei(self, raw_records):
        df = records_to_frame(raw_records)
        assert str(df["time"].dt.tz) == "Asia/Taipei"

    def test_unparseable_dates_raise_instead_of_dropping(self, raw_records):
        bad = daily_record("古亭", "PM2.5", "2024/01/02", ["1"] * 24)
        with pytest.raises(ValueError, match="2024/01/02"):
            records_to_frame([*raw_records, bad])

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing"):
            records_to_frame([{"sitename": "古亭", "monitordate": "2024-01-01"}])

    def test_empty_records_raise(self):
        with pytest.raises(ValueError):
            records_to_frame([])


class TestFillShortGaps:
    def test_fills_gap_up_to_limit(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])
        out = fill_short_gaps(s, max_gap=3)
        np.testing.assert_allclose(out, [1, 2, 3, 4, 5])

    def test_leaves_long_gap_entirely_missing(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, 6.0])
        out = fill_short_gaps(s, max_gap=3)
        assert out.isna().sum() == 4

    def test_does_not_extrapolate_edges(self):
        s = pd.Series([np.nan, 1.0, 2.0, np.nan])
        out = fill_short_gaps(s, max_gap=3)
        assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[-1])

    def test_does_not_mutate_input(self):
        s = pd.Series([1.0, np.nan, 3.0])
        fill_short_gaps(s, max_gap=3)
        assert np.isnan(s.iloc[1])


class TestEncodeWindDirection:
    def test_sin_cos_encoding(self):
        df = pd.DataFrame({"wind_dir": [0.0, 90.0, 180.0, np.nan]})
        out = encode_wind_direction(df)
        np.testing.assert_allclose(out["wind_sin"].iloc[:3], [0, 1, 0], atol=1e-9)
        np.testing.assert_allclose(out["wind_cos"].iloc[:3], [1, 0, -1], atol=1e-9)
        assert np.isnan(out["wind_sin"].iloc[3])
        assert "wind_dir" not in out.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"wind_dir": [0.0]})
        encode_wind_direction(df)
        assert list(df.columns) == ["wind_dir"]


class TestPreprocessStation:
    def _station_frame(self, hourly_index):
        n = len(hourly_index)
        return pd.DataFrame({
            "station": "古亭",
            "time": hourly_index,
            "pm25": np.arange(n, dtype=float),
            "pm10": np.ones(n), "no2": np.ones(n), "o3": np.ones(n),
            "wind_speed": np.ones(n), "wind_dir": np.full(n, 350.0),
        })

    def test_reindexes_to_complete_hourly_grid(self, hourly_index):
        df = self._station_frame(hourly_index).drop(index=[5, 6])
        out = preprocess_station(df)
        assert len(out) == len(hourly_index)
        assert out.index.freq == "h" or pd.infer_freq(out.index) == "h"
        assert out["pm25"].iloc[5] == 5.0  # interpolated

    def test_wind_interpolated_in_sin_cos_space(self, hourly_index):
        df = self._station_frame(hourly_index)
        df.loc[1, "wind_dir"] = np.nan
        df.loc[0, "wind_dir"] = 350.0
        df.loc[2, "wind_dir"] = 10.0
        out = preprocess_station(df)
        # 350° and 10° straddle north; correct interpolation stays near north (cos ~ 1)
        assert out["wind_cos"].iloc[1] > 0.9

    def test_rejects_multiple_stations(self, hourly_index):
        df = self._station_frame(hourly_index)
        df.loc[0, "station"] = "板橋"
        with pytest.raises(ValueError, match="single station"):
            preprocess_station(df)
