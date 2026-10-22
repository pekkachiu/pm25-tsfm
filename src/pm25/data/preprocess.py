"""Cleaning, short-gap interpolation and wind encoding for MOENV station data."""

import numpy as np
import pandas as pd

from pm25.config import MAX_INTERP_GAP_HOURS, MEASUREMENTS, RAW_COLUMNS, TIMEZONE


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Convert raw API records to a typed long-format frame; invalid flags become NaN."""
    if not records:
        raise ValueError("no records to convert")
    raw = pd.DataFrame.from_records(records)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")

    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    numeric = {
        col: pd.to_numeric(df[col], errors="coerce").where(lambda v: v >= 0)
        for col in MEASUREMENTS
    }
    times = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize(TIMEZONE)
    out = df.assign(time=times, **numeric).dropna(subset=["time"])
    return (
        out.drop_duplicates(subset=["station", "time"], keep="first")
        .sort_values(["station", "time"])
        .reset_index(drop=True)
    )


def fill_short_gaps(series: pd.Series, max_gap: int = MAX_INTERP_GAP_HOURS) -> pd.Series:
    """Linearly interpolate interior NaN runs of length <= max_gap; longer runs stay NaN."""
    is_nan = series.isna()
    run_id = (~is_nan).cumsum()
    run_len = is_nan.groupby(run_id).transform("sum")
    interpolated = series.interpolate(method="linear", limit_area="inside")
    fillable = is_nan & (run_len <= max_gap)
    return series.where(~fillable, interpolated)


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Replace wind_dir (degrees) with its (sin, cos) encoding."""
    radians = np.deg2rad(df["wind_dir"])
    return df.drop(columns="wind_dir").assign(wind_sin=np.sin(radians), wind_cos=np.cos(radians))


def preprocess_station(df: pd.DataFrame, max_gap: int = MAX_INTERP_GAP_HOURS) -> pd.DataFrame:
    """Build a complete hourly, time-indexed frame for one station with short gaps filled.

    Wind direction is encoded before interpolation so gaps straddling north are handled.
    """
    if df["station"].nunique() != 1:
        raise ValueError("preprocess_station expects a single station")
    station = df["station"].iloc[0]
    indexed = encode_wind_direction(df).drop(columns="station").set_index("time").sort_index()
    grid = pd.date_range(indexed.index.min(), indexed.index.max(), freq="h", name="time")
    reindexed = indexed.reindex(grid)
    filled = reindexed.apply(lambda col: fill_short_gaps(col, max_gap))
    return filled.assign(station=station)
