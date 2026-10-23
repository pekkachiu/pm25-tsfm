import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.analysis import (
    diebold_mariano,
    loss_differential,
    pairwise_tests,
    recovery_times,
    station_wins,
)


def _preds(model, errors, station="古亭", lead=1, start="2025-01-01"):
    n = len(errors)
    origins = pd.date_range(start, periods=n, freq="h", tz="Asia/Taipei")
    y = np.full(n, 20.0)
    return pd.DataFrame({"station": station, "model": model, "origin": origins, "lead": lead,
                         "target_time": origins + pd.Timedelta(hours=lead), "y": y,
                         "q10": y, "q50": y + np.asarray(errors), "q90": y})


def test_loss_differential_aligns_on_station_and_origin():
    a = _preds("a", [1.0, -2.0, 3.0])
    b = _preds("b", [0.5, 0.5, 0.5]).iloc[[2, 0, 1]]  # shuffled
    d = loss_differential(pd.concat([a, b]), "a", "b", lead=1)
    assert np.allclose(d.sort_index(), [0.5, 1.5, 2.5])


def test_loss_differential_pools_stations_by_target_time():
    preds = pd.concat([_preds("a", [2.0, 2.0]), _preds("b", [0.0, 0.0]),
                       _preds("a", [4.0, 4.0], station="小港"),
                       _preds("b", [0.0, 0.0], station="小港")])
    d = loss_differential(preds, "a", "b", lead=1)
    assert len(d) == 2 and np.allclose(d, 3.0)


def test_diebold_mariano_detects_clear_difference():
    rng = np.random.default_rng(0)
    result = diebold_mariano(pd.Series(rng.normal(0.5, 1.0, 2000)), lag=24)
    assert result["mean_diff"] == pytest.approx(0.5, abs=0.1)
    assert result["p_value"] < 0.001 and result["dm_stat"] > 0


def test_diebold_mariano_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    result = diebold_mariano(pd.Series(rng.normal(0.0, 1.0, 2000)), lag=24)
    assert result["p_value"] > 0.05


def test_diebold_mariano_hac_widens_for_autocorrelated_series():
    rng = np.random.default_rng(2)
    shocks = rng.normal(0, 1, 3000)
    persistent = pd.Series(np.convolve(shocks, np.ones(24), mode="same") / 24 + 0.05)
    assert (diebold_mariano(persistent, lag=48)["p_value"]
            > diebold_mariano(persistent, lag=0)["p_value"])


def test_diebold_mariano_too_short_raises():
    with pytest.raises(ValueError, match="observations"):
        diebold_mariano(pd.Series([1.0]), lag=0)


def test_pairwise_tests_table():
    rng = np.random.default_rng(3)
    preds = pd.concat([_preds("a", rng.normal(0, 3, 500)), _preds("b", rng.normal(0, 1, 500))])
    table = pairwise_tests(preds, pairs=(("a", "b"),), leads=(1,))
    row = table.iloc[0]
    assert (row["model"], row["baseline"], row["lead"]) == ("a", "b", 1)
    assert row["p_value"] < 0.05 and row["mean_diff"] > 0


def test_pairwise_tests_skips_missing_models():
    table = pairwise_tests(_preds("a", [1.0] * 10), pairs=(("a", "zzz"),), leads=(1,))
    assert table.empty


def test_station_wins_counts_lower_mae():
    preds = pd.concat([_preds("a", [1.0] * 5), _preds("b", [2.0] * 5),
                       _preds("a", [3.0] * 5, station="小港"),
                       _preds("b", [2.0] * 5, station="小港")])
    wins = station_wins(preds, pairs=(("a", "b"),), leads=(1,)).iloc[0]
    assert (wins["wins"], wins["stations"]) == (1, 2)


def test_recovery_times_pairs_up_and_down_changes():
    t = pd.to_datetime(["2025-01-01 00:00", "2025-01-02 00:00", "2025-01-04 00:00",
                        "2025-01-05 00:00"]).tz_localize("Asia/Taipei")
    drift = pd.DataFrame({"station": "古亭", "model": "a", "lead": 24, "time": t,
                          "mean_error_after": [3.0, 6.0, 2.0, 5.0]})
    out = recovery_times(drift)
    # up at 01-02 (3->6) recovers at 01-04 (6->2): 48 h; up at 01-05 never recovers
    assert list(out["hours"]) == [48.0]
    assert out.iloc[0]["model"] == "a"
