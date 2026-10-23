"""Day-ahead wind forecasts from Open-Meteo's archive of previous model runs.

Run `uv run python -m pm25.data.wind_forecast` to write data/wind_forecast/<站>.csv.gz.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pm25.config import (
    OPEN_METEO_PREVIOUS_RUNS_URL,
    STATION_COORDS,
    TEST_YEAR,
    TIMEZONE,
    WIND_FORECAST_DIR,
    WIND_FORECAST_START,
)
from pm25.data.io import write_station

SPEED = "wind_speed_10m_previous_day1"
DIRECTION = "wind_direction_10m_previous_day1"
REQUEST_TIMEOUT_SECONDS = 120


def request_params(latitude: float, longitude: float, start: str, end: str) -> dict:
    return {
        "latitude": latitude, "longitude": longitude,
        "hourly": f"{SPEED},{DIRECTION}",
        "start_date": start, "end_date": end,
        "wind_speed_unit": "ms", "timezone": TIMEZONE,
    }


def parse_forecast(payload: dict) -> pd.DataFrame:
    """Hourly forecast frame with speed and (sin, cos) direction, indexed in local time."""
    hourly = payload.get("hourly")
    if not hourly or SPEED not in hourly or DIRECTION not in hourly:
        raise ValueError(f"no hourly wind forecast in response: {payload.get('reason', payload)}")
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], format="%Y-%m-%dT%H:%M"),
                             name="time").tz_localize(TIMEZONE)
    speed = pd.to_numeric(pd.Series(hourly[SPEED], dtype=object), errors="coerce").to_numpy()
    direction = pd.to_numeric(pd.Series(hourly[DIRECTION], dtype=object),
                              errors="coerce").to_numpy()
    radians = np.deg2rad(direction)
    return pd.DataFrame({"fc_wind_speed": speed, "fc_wind_sin": np.sin(radians),
                         "fc_wind_cos": np.cos(radians)}, index=index)


def download_station(station: str, start: str, end: str, session=None) -> pd.DataFrame:
    if station not in STATION_COORDS:
        raise ValueError(f"no coordinates configured for station {station!r}")
    session = session or requests.Session()
    response = session.get(OPEN_METEO_PREVIOUS_RUNS_URL,
                           params=request_params(*STATION_COORDS[station], start, end),
                           timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_forecast(response.json())


def attach_wind_forecast(df: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Station frame plus forecast columns; hours without a forecast stay NaN."""
    return df.join(forecast.reindex(df.index))


def main() -> None:  # pragma: no cover - network entry point
    out = Path(WIND_FORECAST_DIR)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for station in STATION_COORDS:
        frame = download_station(station, WIND_FORECAST_START, f"{TEST_YEAR}-12-31", session)
        print(station, write_station(frame, out, station), frame.isna().mean().round(4).to_dict())


if __name__ == "__main__":  # pragma: no cover
    main()
