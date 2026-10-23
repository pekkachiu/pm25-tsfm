import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.conformal import adaptive_intervals


def _stream(n=2000, lead=1, half_width=0.5, station="古亭", model="m", seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-01-01", periods=n, freq="h", tz="Asia/Taipei")
    center = np.full(n, 20.0)
    y = center + rng.normal(shift, 1.0, n)
    return pd.DataFrame({"station": station, "model": model, "origin": origins, "lead": lead,
                         "target_time": origins + pd.Timedelta(hours=lead), "y": y,
                         "q10": center - half_width, "q50": center,
                         "q90": center + half_width})


def _coverage(df):
    return ((df["y"] >= df["q10"]) & (df["y"] <= df["q90"])).mean()


def test_too_narrow_intervals_are_widened_to_target_coverage():
    raw = _stream(n=6000, half_width=0.5)  # covers about 38%
    out = adaptive_intervals(raw, alpha=0.2, step=0.05)
    assert _coverage(raw) < 0.45
    assert _coverage(out.iloc[1000:]) == pytest.approx(0.8, abs=0.03)


def test_too_wide_intervals_are_narrowed():
    raw = _stream(n=6000, half_width=4.0)  # covers almost everything
    out = adaptive_intervals(raw, alpha=0.2, step=0.05)
    assert _coverage(out.iloc[1000:]) == pytest.approx(0.8, abs=0.03)
    assert (out["q90"] - out["q10"]).iloc[1000:].mean() < 8.0


def test_no_lookahead_until_targets_are_observed():
    raw = _stream(n=200, lead=24)
    out = adaptive_intervals(raw, alpha=0.2, step=1.0)
    # targets of the first 24 origins are only observed from origin 24 on
    first = out.iloc[:24]
    assert np.allclose(first["q10"], raw["q10"].iloc[:24])
    assert np.allclose(first["q90"], raw["q90"].iloc[:24])
    assert not np.allclose(out["q90"].iloc[30:], raw["q90"].iloc[30:])


def test_one_sided_miscalibration_moves_only_that_side():
    raw = _stream(n=6000, half_width=1.28, shift=1.5)  # observations biased upward
    out = adaptive_intervals(raw, alpha=0.2, step=0.05)
    tail = out.iloc[1000:]
    assert (tail["q90"] - raw["q90"].iloc[1000:]).mean() > 1.0
    assert (tail["q10"] - raw["q10"].iloc[1000:]).mean() > 0.5  # lower bound moves up


def test_groups_are_independent_and_input_unchanged():
    a = _stream(n=500, station="古亭", half_width=0.5)
    b = _stream(n=500, station="小港", half_width=4.0, seed=1)
    both = pd.concat([a, b], ignore_index=True)
    before = both.copy()
    out = adaptive_intervals(both, alpha=0.2, step=0.05)
    pd.testing.assert_frame_equal(both, before)
    alone = adaptive_intervals(a, alpha=0.2, step=0.05)
    assert np.allclose(out.loc[out["station"] == "古亭", "q90"].to_numpy(),
                       alone["q90"].to_numpy())
    assert list(out.index) == list(both.index)


def test_interval_never_excludes_median():
    raw = _stream(n=3000, half_width=6.0)
    out = adaptive_intervals(raw, alpha=0.2, step=0.5)
    assert (out["q10"] <= out["q50"]).all() and (out["q90"] >= out["q50"]).all()


def test_missing_observations_do_not_update():
    raw = _stream(n=100, lead=1)
    raw.loc[:, "y"] = np.nan
    out = adaptive_intervals(raw, alpha=0.2, step=1.0)
    assert np.allclose(out["q10"], raw["q10"]) and np.allclose(out["q90"], raw["q90"])
