import numpy as np
import pandas as pd
import pytest

from pm25.data.preprocess import (
    encode_wind_direction,
    fill_short_gaps,
    preprocess_station,
    records_to_frame,
)


class TestRecordsToFrame:
    def test_parses_numeric_and_marks_invalid_as_nan(self, raw_records):
        df = records_to_frame(raw_records)
        guting = df[df["station"] == "古亭"].sort_values("time")
        assert np.isnan(guting["pm25"].iloc[0])  # "x" flag
        assert guting["pm25"].iloc[1] == 12.0
        assert np.isnan(guting["no2"].iloc[0])  # "-" flag

    def test_negative_and_empty_values_become_nan(self, raw_records):
        df = records_to_frame(raw_records)
        banqiao = df[df["station"] == "板橋"].iloc[0]
        assert np.isnan(banqiao["pm25"])
        assert np.isnan(banqiao["pm10"])
        assert np.isnan(banqiao["o3"])

    def test_time_is_tz_aware_taipei(self, raw_records):
        df = records_to_frame(raw_records)
        assert str(df["time"].dt.tz) == "Asia/Taipei"

    def test_renames_columns_to_canonical(self, raw_records):
        df = records_to_frame(raw_records)
        for col in ["station", "time", "pm25", "pm10", "no2", "o3", "wind_speed", "wind_dir"]:
            assert col in df.columns

    def test_drops_duplicate_station_hours(self, raw_records):
        df = records_to_frame(raw_records + [raw_records[0]])
        assert len(df) == 3

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing"):
            records_to_frame([{"sitename": "古亭", "datacreationdate": "2024-01-01 00:00"}])

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
