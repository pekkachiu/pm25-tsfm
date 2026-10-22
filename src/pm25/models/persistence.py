"""Persistence baseline: y_hat[t+h] = y[t]."""

import numpy as np

from pm25.config import QUANTILE_LEVELS, TARGET
from pm25.evaluation.rolling import ForecastInput


class PersistenceModel:
    name = "persistence"

    def predict(self, inputs: list[ForecastInput], horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        """Return forecasts of shape (n_inputs, n_quantiles, horizon)."""
        last = np.array([x.context[TARGET].iloc[-1] for x in inputs], dtype=float)
        if np.isnan(last).any():
            raise ValueError("last context value is NaN")
        return np.broadcast_to(last[:, None, None],
                               (len(inputs), len(quantile_levels), horizon)).copy()
