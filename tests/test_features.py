import numpy as np
import pandas as pd
import pytest

from pm25.models.features import LAGS, feature_matrix, training_set


def _frame(n=24 * 6):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="Asia/Taipei")
    return pd.DataFrame({
        "pm25": np.arange(n, dtype=float),
        "wind_speed": np.full(n, 2.0),
        "no2": np.arange(n, dtype=float) * 0.5,
    }, index=idx)


def test_feature_matrix_lags_and_calendar():
    df = _frame()
    tails = [df.iloc[i - LAGS + 1:i + 1] for i in (LAGS - 1, LAGS + 5)]
    X = feature_matrix(tails, covariates=("wind_speed", "no2"))
    assert len(X) == 2
    assert X["pm25_lag0"].tolist() == [47.0, 53.0]
    assert X["pm25_lag47"].tolist() == [0.0, 6.0]
    assert X["pm25_mean24"].iloc[0] == pytest.approx(np.mean(np.arange(24, 48)))
    assert X["no2_mean24"].iloc[0] == pytest.approx(np.mean(np.arange(24, 48)) * 0.5)
    assert X["wind_speed_lag5"].iloc[0] == 2.0
    assert X["hour"].iloc[1] == tails[1].index[-1].hour
    assert {"dayofweek", "month"} <= set(X.columns)


def test_feature_matrix_requires_enough_history():
    df = _frame()
    with pytest.raises(ValueError, match="48"):
        feature_matrix([df.iloc[:10]], covariates=())


def test_feature_matrix_keeps_nan_for_trees():
    df = _frame()
    df.iloc[10, 0] = np.nan
    X = feature_matrix([df.iloc[:LAGS]], covariates=())
    assert X.isna().any(axis=None)


def test_training_set_aligns_targets_with_lead():
    df = _frame()
    X, y = training_set(df, lead=6, covariates=("wind_speed",))
    # first usable origin has 48h history; target is pm25 six hours later
    assert X["pm25_lag0"].iloc[0] == 47.0
    assert y.iloc[0] == 53.0
    assert len(X) == len(df) - LAGS + 1 - 6


def test_training_set_drops_missing_targets():
    df = _frame()
    df.iloc[60, 0] = np.nan
    X, y = training_set(df, lead=1, covariates=())
    assert not y.isna().any()
    assert len(X) == len(y)
