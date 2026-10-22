import numpy as np
import pandas as pd
import pytest

from pm25.data.split import split_by_year
from pm25.models.persistence import PersistenceModel


def test_split_by_year():
    idx = pd.date_range("2023-12-31 22:00", periods=6, freq="h", tz="Asia/Taipei")
    df = pd.DataFrame({"pm25": range(6)}, index=idx)
    train, test = split_by_year(df, test_year=2024)
    assert train.index.max() < pd.Timestamp("2024-01-01", tz="Asia/Taipei")
    assert (test.index.year == 2024).all()
    assert len(train) + len(test) == 6


def test_split_by_year_empty_test_raises():
    idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="Asia/Taipei")
    with pytest.raises(ValueError):
        split_by_year(pd.DataFrame({"pm25": range(3)}, index=idx), test_year=2025)


def test_persistence_repeats_last_value_for_all_quantiles():
    ctx = np.array([3.0, 4.0, 7.0])
    q = PersistenceModel().predict(ctx, horizon=4, quantile_levels=(0.1, 0.5, 0.9))
    assert q.shape == (3, 4)
    assert np.all(q == 7.0)


def test_persistence_rejects_nan_last_value():
    with pytest.raises(ValueError):
        PersistenceModel().predict(np.array([1.0, np.nan]), horizon=1)
