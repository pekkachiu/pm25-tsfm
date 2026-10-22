"""Rolling-origin evaluation windows."""

from dataclasses import dataclass

import pandas as pd

from pm25.config import ORIGIN_HOURS


@dataclass(frozen=True)
class Window:
    origin: pd.Timestamp  # last observed hour; forecasts cover origin+1h .. origin+horizon
    context: pd.Series
    target: pd.Series


def make_windows(series: pd.Series, context_length: int, horizon: int,
                 origin_hours: tuple[int, ...] = ORIGIN_HOURS,
                 start=None, end=None) -> list[Window]:
    """Build windows on an hourly series, dropping any whose context or target has NaN.

    start/end restrict the origins only; context may extend before start.
    """
    if context_length < 1 or horizon < 1:
        raise ValueError("context_length and horizon must be >= 1")
    index = series.index
    lo = pd.Timestamp(start, tz=index.tz) if start is not None else index[0]
    hi = pd.Timestamp(end, tz=index.tz) if end is not None else index[-1]
    positions = [
        i for i, ts in enumerate(index)
        if lo <= ts <= hi and ts.hour in origin_hours
        and i >= context_length - 1 and i + horizon < len(index)
    ]
    windows = (
        Window(origin=index[i],
               context=series.iloc[i - context_length + 1:i + 1],
               target=series.iloc[i + 1:i + 1 + horizon])
        for i in positions
    )
    return [w for w in windows if not (w.context.isna().any() or w.target.isna().any())]
