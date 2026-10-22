"""Persistence baseline: y_hat[t+h] = y[t]."""

import numpy as np

from pm25.config import QUANTILE_LEVELS


class PersistenceModel:
    name = "persistence"

    def predict(self, context: np.ndarray, horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        """Return forecasts of shape (len(quantile_levels), horizon)."""
        last = float(np.asarray(context, dtype=float)[-1])
        if np.isnan(last):
            raise ValueError("last context value is NaN")
        return np.full((len(quantile_levels), horizon), last)
