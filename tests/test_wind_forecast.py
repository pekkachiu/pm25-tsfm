from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pm25.config import STATION_COORDS, STATIONS, WIND_FORECAST_COVARIATES
from pm25.data.wind_forecast import (
    attach_wind_forecast,
    download_station,
    parse_forecast,
    request_params,
)


def _payload(n=3, speed=None, direction=None):
    times = pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M")
    return {"hourly": {
        "time": list(times),
        "wind_speed_10m_previous_day1": speed if speed is not None else [1.0] * n,
        "wind_direction_10m_previous_day1": direction if direction is not None else [90.0] * n,
    }}


def test_every_station_has_coordinates():
    assert {s for g in STATIONS.values() for s in g} == set(STATION_COORDS)


def test_request_params_use_day_ahead_forecast():
    params = request_params(25.0, 121.5, "2024-12-01", "2025-12-31")
    assert params["latitude"] == 25.0 and params["longitude"] == 121.5
    assert "wind_speed_10m_previous_day1" in params["hourly"]
    assert params["wind_speed_unit"] == "ms" and params["timezone"] == "Asia/Taipei"


def test_parse_forecast_encodes_direction_and_localizes():
    out = parse_forecast(_payload(direction=[0.0, 90.0, 180.0]))
    assert list(out.columns) == list(WIND_FORECAST_COVARIATES)
    assert str(out.index.tz) == "Asia/Taipei"
    assert out.index[0] == pd.Timestamp("2025-01-01 00:00", tz="Asia/Taipei")
    assert np.allclose(out["fc_wind_sin"], [0, 1, 0], atol=1e-9)
    assert np.allclose(out["fc_wind_cos"], [1, 0, -1], atol=1e-9)


def test_parse_forecast_null_becomes_nan():
    out = parse_forecast(_payload(speed=[1.0, None, 2.0]))
    assert np.isnan(out["fc_wind_speed"].iloc[1])


def test_parse_forecast_rejects_error_payload():
    with pytest.raises(ValueError, match="hourly"):
        parse_forecast({"error": True, "reason": "bad"})


def test_download_station_calls_api_with_station_coords():
    response = MagicMock()
    response.json.return_value = _payload()
    session = MagicMock()
    session.get.return_value = response
    out = download_station("古亭", "2025-01-01", "2025-01-01", session=session)
    params = session.get.call_args.kwargs["params"]
    assert (params["latitude"], params["longitude"]) == STATION_COORDS["古亭"]
    assert len(out) == 3


def test_download_station_unknown_station_raises():
    with pytest.raises(ValueError, match="coordinates"):
        download_station("不存在", "2025-01-01", "2025-01-01", session=MagicMock())


def test_attach_wind_forecast_aligns_without_mutating():
    idx = pd.date_range("2024-12-31 23:00", periods=3, freq="h", tz="Asia/Taipei")
    df = pd.DataFrame({"pm25": [1.0, 2.0, 3.0]}, index=idx)
    out = attach_wind_forecast(df, parse_forecast(_payload(n=2)))
    assert list(df.columns) == ["pm25"]
    assert np.isnan(out["fc_wind_speed"].iloc[0])
    assert out["fc_wind_speed"].iloc[1] == 1.0
    assert out.index.equals(df.index)
