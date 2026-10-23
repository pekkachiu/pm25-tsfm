"""Day-ahead wind forecasts from Open-Meteo's archive of previous model runs.

Run `uv run python -m pm25.data.wind_forecast [--model jma_gsm|best_match]` to write
data/wind_forecast/<model>/<站>.csv.gz.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pm25.config import (
    NWP_MODELS,
    OPEN_METEO_PREVIOUS_RUNS_URL,
    PRIMARY_NWP,
    STATION_COORDS,
    TEST_YEAR,
    TIMEZONE,
    WIND_FORECAST_DIR,
)
from pm25.data.io import write_station

SPEED = "wind_speed_10m_previous_day1"
DIRECTION = "wind_direction_10m_previous_day1"
REQUEST_TIMEOUT_SECONDS = 120


def forecast_dir(model: str) -> Path:
    return Path(WIND_FORECAST_DIR) / model


def request_params(latitude: float, longitude: float, start: str, end: str,
                   model: str = PRIMARY_NWP) -> dict:
    if model not in NWP_MODELS:
        raise ValueError(f"unknown NWP model {model!r}; choose from {sorted(NWP_MODELS)}")
    params = {
        "latitude": latitude, "longitude": longitude,
        "hourly": f"{SPEED},{DIRECTION}",
        "start_date": start, "end_date": end,
        "wind_speed_unit": "ms", "timezone": TIMEZONE,
    }
    # best_match lets Open-Meteo pick (and change) the model, so it is never requested by name.
    return params if model == "best_match" else {**params, "models": model}


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


def download_station(station: str, start: str, end: str, session=None,
                     model: str = PRIMARY_NWP) -> pd.DataFrame:
    if station not in STATION_COORDS:
        raise ValueError(f"no coordinates configured for station {station!r}")
    session = session or requests.Session()
    response = session.get(OPEN_METEO_PREVIOUS_RUNS_URL,
                           params=request_params(*STATION_COORDS[station], start, end, model),
                           timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_forecast(response.json())


def attach_wind_forecast(df: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Station frame plus forecast columns; hours without a forecast stay NaN."""
    return df.join(forecast.reindex(df.index))


def wind_forecast_skill(observed: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Agreement of forecast with observed wind over hours where both speeds exist."""
    both = observed.join(forecast, how="inner").dropna(subset=["wind_speed", "fc_wind_speed"])
    cross = both["wind_sin"] * both["fc_wind_cos"] - both["wind_cos"] * both["fc_wind_sin"]
    dot = both["wind_sin"] * both["fc_wind_sin"] + both["wind_cos"] * both["fc_wind_cos"]
    return {"corr": both["wind_speed"].corr(both["fc_wind_speed"]),
            "obs_mean": both["wind_speed"].mean(), "fc_mean": both["fc_wind_speed"].mean(),
            "dir_error": float(np.abs(np.degrees(np.arctan2(cross, dot))).median()),
            "hours": len(both)}


def main() -> None:  # pragma: no cover - network entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(NWP_MODELS), default=PRIMARY_NWP)
    model = parser.parse_args().model
    out = forecast_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for station in STATION_COORDS:
        frame = download_station(station, NWP_MODELS[model], f"{TEST_YEAR}-12-31", session,
                                 model)
        print(station, write_station(frame, out, station), frame.isna().mean().round(4).to_dict())


if __name__ == "__main__":  # pragma: no cover
    main()
