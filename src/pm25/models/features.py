"""Tabular features for the trained baselines, computed from the last LAGS hours.

Training and prediction share `_features`, so both see identical feature definitions.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pm25.config import TARGET

LAGS = 48  # PM2.5 lags 1-48 h (lag0 = value at the origin)
ROLLING_WINDOWS = (6, 24)
COVARIATE_LAGS = 6


def _features(target_hist: np.ndarray, cov_hists: dict[str, np.ndarray],
              origins: pd.DatetimeIndex) -> pd.DataFrame:
    """target_hist and each cov_hist have shape (n, LAGS), oldest -> newest."""
    newest_first = target_hist[:, ::-1]
    columns = {f"{TARGET}_lag{k}": newest_first[:, k] for k in range(LAGS)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN
        for w in ROLLING_WINDOWS:
            recent = target_hist[:, -w:]
            columns |= {f"{TARGET}_mean{w}": np.nanmean(recent, axis=1),
                        f"{TARGET}_std{w}": np.nanstd(recent, axis=1),
                        f"{TARGET}_min{w}": np.nanmin(recent, axis=1),
                        f"{TARGET}_max{w}": np.nanmax(recent, axis=1)}
        for name, hist in cov_hists.items():
            columns |= {f"{name}_lag{k}": hist[:, ::-1][:, k] for k in range(COVARIATE_LAGS)}
            columns[f"{name}_mean24"] = np.nanmean(hist[:, -24:], axis=1)
    columns |= {"hour": origins.hour, "dayofweek": origins.day_of_week, "month": origins.month}
    return pd.DataFrame(columns, index=origins)


def feature_matrix(tails: list[pd.DataFrame], covariates: tuple[str, ...]) -> pd.DataFrame:
    """One feature row per context frame, using its last LAGS hours."""
    if any(len(t) < LAGS for t in tails):
        raise ValueError(f"each context needs at least {LAGS} hours")
    recent = [t.iloc[-LAGS:] for t in tails]
    target_hist = np.stack([t[TARGET].to_numpy(dtype=float) for t in recent])
    cov_hists = {c: np.stack([t[c].to_numpy(dtype=float) for t in recent]) for c in covariates}
    origins = pd.DatetimeIndex([t.index[-1] for t in recent])
    return _features(target_hist, cov_hists, origins)


def _future_columns(name: str, at_lead, mean_to_lead) -> dict:
    return {f"{name}_at_lead": at_lead, f"{name}_mean_to_lead": mean_to_lead}


def future_features(futures: list[pd.DataFrame], lead: int,
                    future_covariates: tuple[str, ...],
                    origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Known-future covariates (e.g. NWP wind) for one lead: the value at origin+lead and
    the mean over origin+1..origin+lead. Each frame covers the forecast hours."""
    columns = {}
    for name in future_covariates:
        values = np.stack([f[name].to_numpy(dtype=float)[:lead] for f in futures])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            columns |= _future_columns(name, values[:, lead - 1], np.nanmean(values, axis=1))
    return pd.DataFrame(columns, index=origins)


def training_future_features(df: pd.DataFrame, origins: pd.DatetimeIndex, lead: int,
                             future_covariates: tuple[str, ...]) -> pd.DataFrame:
    """Same definition as `future_features`, computed for every training origin."""
    columns = {}
    for name in future_covariates:
        series = df[name]
        running = series.rolling(lead, min_periods=1).mean().shift(-lead)
        columns |= _future_columns(name, series.shift(-lead).loc[origins].to_numpy(),
                                   running.loc[origins].to_numpy())
    return pd.DataFrame(columns, index=origins)


def training_set(df: pd.DataFrame, lead: int, covariates: tuple[str, ...],
                 future_covariates: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Features at every origin with LAGS hours of history and the target `lead` hours ahead."""
    def windows(column: str) -> np.ndarray:
        return sliding_window_view(df[column].to_numpy(dtype=float), LAGS)[:-lead]

    origins = df.index[LAGS - 1:len(df) - lead]
    X = _features(windows(TARGET), {c: windows(c) for c in covariates}, origins)
    if future_covariates:
        X = X.join(training_future_features(df, origins, lead, future_covariates))
    y = pd.Series(df[TARGET].to_numpy(dtype=float)[LAGS - 1 + lead:], index=origins)
    keep = y.notna().to_numpy()
    return X[keep], y[keep]
