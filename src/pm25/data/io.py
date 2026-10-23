"""Version-independent storage (gzipped CSV) for processed data and predictions.

Pickles written by pandas 3 cannot be read by pandas 2 (e.g. on Colab), so CSV is used.
"""

from pathlib import Path

import pandas as pd

from pm25.config import TIMEZONE

PREDICTION_TIME_COLUMNS = ("origin", "target_time")


def _parse_times(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(TIMEZONE)


def write_station(df: pd.DataFrame, directory: Path, station: str) -> Path:
    path = Path(directory) / f"{station}.csv.gz"
    df.to_csv(path, index_label="time")
    return path


def read_station(directory: Path, station: str) -> pd.DataFrame:
    df = pd.read_csv(Path(directory) / f"{station}.csv.gz")
    index = pd.DatetimeIndex(_parse_times(df.pop("time")), name="time")
    return df.set_index(index)


def write_predictions(preds: pd.DataFrame, path: Path) -> None:
    preds.to_csv(path, index=False)


def read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.assign(**{c: _parse_times(df[c]) for c in PREDICTION_TIME_COLUMNS})
