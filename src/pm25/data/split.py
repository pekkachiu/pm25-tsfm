"""Chronological train/test split by calendar year."""

import pandas as pd


def split_by_year(df: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rows before test_year, rows in test_year)."""
    years = df.index.year
    test = df[years == test_year]
    if test.empty:
        raise ValueError(f"no data for test year {test_year}")
    return df[years < test_year], test
