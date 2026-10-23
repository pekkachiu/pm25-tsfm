"""Adaptive conformal inference (ACI) for the q10-q90 forecast intervals.

Each (station, model, lead) stream keeps a lower and an upper margin. After an outcome is
observed, a margin grows by step * (1 - alpha/2) if that side missed and shrinks by
step * alpha/2 otherwise, so each side's long-run miss rate tends to alpha/2 (online
quantile tracking; Gibbs & Candes, 2021). A forecast issued at origin t only uses outcomes
whose target time is <= t, i.e. a lead-h forecast is corrected with an h-hour delay.
"""

import numpy as np
import pandas as pd

from pm25.config import ACI_STEP

GROUP = ["station", "model", "lead"]


def _track(origin: np.ndarray, target: np.ndarray, y: np.ndarray, lo: np.ndarray,
           mid: np.ndarray, hi: np.ndarray, alpha: float,
           step: float) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted bounds for one stream sorted by origin (target = origin + lead)."""
    n = len(y)
    new_lo, new_hi = lo.copy(), hi.copy()
    m_lo = m_hi = 0.0
    observed = 0  # forecasts whose outcome is known at the current origin
    for i in range(n):
        while observed < n and target[observed] <= origin[i]:
            j = observed
            if not np.isnan(y[j]):
                m_lo += step * (float(y[j] < new_lo[j]) - alpha / 2)
                m_hi += step * (float(y[j] > new_hi[j]) - alpha / 2)
            observed += 1
        # the median always stays inside the interval
        new_lo[i] = min(lo[i] - m_lo, mid[i])
        new_hi[i] = max(hi[i] + m_hi, mid[i])
    return new_lo, new_hi


def adaptive_intervals(preds: pd.DataFrame, alpha: float = 0.2,
                       step: float = ACI_STEP) -> pd.DataFrame:
    """Copy of `preds` with q10/q90 replaced by ACI-adjusted bounds (same row order)."""
    q10 = preds["q10"].to_numpy(dtype=float).copy()
    q90 = preds["q90"].to_numpy(dtype=float).copy()
    origin = preds["origin"].to_numpy(dtype="datetime64[ns]").astype("int64")
    target = preds["target_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    y, q50 = preds["y"].to_numpy(dtype=float), preds["q50"].to_numpy(dtype=float)
    for rows in preds.groupby(GROUP, sort=False).indices.values():
        order = rows[np.argsort(origin[rows], kind="stable")]
        q10[order], q90[order] = _track(origin[order], target[order], y[order], q10[order],
                                        q50[order], q90[order], alpha, step)
    return preds.assign(q10=q10, q90=q90)
