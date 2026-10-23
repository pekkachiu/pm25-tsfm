"""Cleaning, short-gap interpolation and wind encoding for MOENV station data."""

import numpy as np
import pandas as pd

from pm25.config import ITEMS, MAX_INTERP_GAP_HOURS, MEASUREMENTS, TIMEZONE

HOUR_COLUMNS = [f"monitorvalue{h:02d}" for h in range(24)]
REQUIRED_COLUMNS = ["sitename", "itemengname", "monitordate", *HOUR_COLUMNS]


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Convert aqx_p_13 daily records to a wide hourly frame; invalid flags become NaN.

    monitorvalueHH on day D is the hour ending at D (HH+1):00, matching aqx_p_488 stamps.
    """
    if not records:
        raise ValueError("no records to convert")
    raw = pd.DataFrame.from_records(records)
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")

    relevant = raw[raw["itemengname"].isin(ITEMS)]
    long = relevant.melt(id_vars=["sitename", "itemengname", "monitordate"],
                         value_vars=HOUR_COLUMNS, var_name="hour", value_name="value")
    dates = long["monitordate"].astype(str).str.strip()
    day = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    unparsed = dates[day.isna()].unique()
    if unparsed.size:
        raise ValueError(f"unparseable monitordate values: {list(unparsed[:5])}")
    hour_ending = pd.to_timedelta(long["hour"].str[-2:].astype(int) + 1, unit="h")
    value = pd.to_numeric(long["value"], errors="coerce")
    tidy = pd.DataFrame({
        "station": long["sitename"],
        "time": (day + hour_ending).dt.tz_localize(TIMEZONE),
        "item": long["itemengname"].map(ITEMS),
        "value": value.where(value >= 0),
    })
    # Duplicate records never disagree on non-null values, so keep the first non-null.
    wide = tidy.pivot_table(index=["station", "time"], columns="item", values="value",
                            aggfunc="first", dropna=False)
    return (wide.reindex(columns=list(MEASUREMENTS)).reset_index()
            .rename_axis(columns=None).sort_values(["station", "time"]).reset_index(drop=True))


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
