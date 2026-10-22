import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.rolling import make_windows


@pytest.fixture
def series():
    idx = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="Asia/Taipei")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_one_origin_per_day_at_given_hour(series):
    wins = make_windows(series, context_length=48, horizon=24, origin_hours=(0,))
    origins = [w.origin for w in wins]
    assert all(o.hour == 0 for o in origins)
    # need 48h context (origin at index >= 47) and 24h target after origin
    assert origins[0] == pd.Timestamp("2025-01-03 00:00", tz="Asia/Taipei")
    assert origins[-1] == pd.Timestamp("2025-01-09 00:00", tz="Asia/Taipei")


def test_window_contents(series):
    w = make_windows(series, context_length=48, horizon=6, origin_hours=(0,))[0]
    assert len(w.context) == 48 and len(w.target) == 6
    assert w.context.index[-1] == w.origin
    assert w.target.iloc[0] == w.context.iloc[-1] + 1


def test_drops_windows_with_missing_values(series):
    s = series.copy()
    s.loc["2025-01-05 03:00"] = np.nan
    wins = make_windows(s, context_length=48, horizon=24, origin_hours=(0,))
    affected = {pd.Timestamp(d, tz="Asia/Taipei") for d in
                ["2025-01-05 00:00", "2025-01-06 00:00", "2025-01-07 00:00"]}
    assert affected.isdisjoint({w.origin for w in wins})


def test_restricts_origins_to_period(series):
    wins = make_windows(series, context_length=24, horizon=1, origin_hours=(0, 12),
                        start="2025-01-05", end="2025-01-05 23:00")
    assert [w.origin.hour for w in wins] == [0, 12]


def test_context_may_reach_before_period(series):
    wins = make_windows(series, context_length=96, horizon=1, origin_hours=(0,),
                        start="2025-01-05")
    assert wins[0].origin == pd.Timestamp("2025-01-05", tz="Asia/Taipei")


def test_invalid_args_raise(series):
    with pytest.raises(ValueError):
        make_windows(series, context_length=0, horizon=1)
    with pytest.raises(ValueError):
        make_windows(series, context_length=1, horizon=0)


def test_required_recent_hours_allows_older_context_gaps(series):
    s = series.copy()
    s.loc["2025-01-05 03:00"] = np.nan
    wins = make_windows(s, context_length=48, horizon=24, origin_hours=(0,),
                        required_recent_hours=12)
    origins = {w.origin for w in wins}
    # gap is 45h before 2025-01-07 00:00: outside the last 12h, so the window is kept
    assert pd.Timestamp("2025-01-07", tz="Asia/Taipei") in origins
    # gap falls inside the target of 2025-01-05 00:00: still dropped
    assert pd.Timestamp("2025-01-05", tz="Asia/Taipei") not in origins


def test_required_recent_hours_drops_recent_gaps(series):
    s = series.copy()
    s.loc["2025-01-06 20:00"] = np.nan
    wins = make_windows(s, context_length=48, horizon=1, origin_hours=(0,),
                        required_recent_hours=12)
    assert pd.Timestamp("2025-01-07", tz="Asia/Taipei") not in {w.origin for w in wins}


def test_required_recent_hours_must_fit_context(series):
    with pytest.raises(ValueError):
        make_windows(series, context_length=24, horizon=1, required_recent_hours=48)
