import numpy as np
import pytest

from pm25.evaluation.metrics import exceedance_scores, interval_coverage, mae, rmse


def test_mae_and_rmse():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 5.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))


def test_metrics_ignore_nan_pairs():
    y = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 100.0, 4.0])
    assert mae(y, p) == pytest.approx(0.5)


def test_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae(np.ones(3), np.ones(2))


def test_metrics_all_nan_returns_nan():
    assert np.isnan(mae(np.array([np.nan]), np.array([1.0])))


def test_interval_coverage():
    y = np.array([1.0, 5.0, 10.0, 3.0])
    lo = np.array([0.0, 0.0, 0.0, 3.0])
    hi = np.array([2.0, 4.0, 20.0, 3.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(0.75)


def test_exceedance_scores():
    y = np.array([10.0, 40.0, 50.0, 20.0, 35.0])
    p = np.array([40.0, 45.0, 10.0, 20.0, 31.0])
    s = exceedance_scores(y, p, threshold=30.5)
    # TP: idx1, idx4; FP: idx0; FN: idx2
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["support"] == 3


def test_exceedance_no_positives_gives_zero_not_error():
    s = exceedance_scores(np.array([1.0, 2.0]), np.array([1.0, 2.0]), threshold=30.5)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0
