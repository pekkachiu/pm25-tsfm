import numpy as np
import pandas as pd

from pm25.data.io import read_predictions, read_station, write_predictions, write_station


def test_station_round_trip_keeps_tz_index_and_nan(tmp_path):
    idx = pd.date_range("2024-12-31 23:00", periods=3, freq="h", tz="Asia/Taipei", name="time")
    df = pd.DataFrame({"pm25": [1.0, np.nan, 3.5], "wind_sin": [0.1, 0.2, 0.3],
                       "station": "古亭"}, index=idx)
    path = write_station(df, tmp_path, "古亭")
    assert path.name == "古亭.csv.gz"
    back = read_station(tmp_path, "古亭")
    pd.testing.assert_frame_equal(back, df, check_freq=False)


def test_predictions_round_trip_keeps_timestamps(tmp_path):
    origins = pd.date_range("2025-01-01", periods=2, freq="h", tz="Asia/Taipei")
    preds = pd.DataFrame({"station": "古亭", "model": "m", "origin": origins, "lead": 1,
                          "target_time": origins + pd.Timedelta(hours=1),
                          "y": [1.0, 2.0], "q10": [0.5, 1.5], "q50": [1.0, 2.0],
                          "q90": [1.5, 2.5]})
    path = tmp_path / "古亭__m.csv.gz"
    write_predictions(preds, path)
    back = read_predictions(path)
    pd.testing.assert_frame_equal(back, preds)
