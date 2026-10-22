import numpy as np
import pandas as pd
import pytest

from pm25.data.split import split_by_year


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
