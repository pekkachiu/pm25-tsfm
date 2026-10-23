"""XGBoost baseline: one multi-quantile model per lead time (direct strategy)."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pm25.config import COVARIATES, HORIZONS, QUANTILE_LEVELS
from pm25.evaluation.rolling import ForecastInput
from pm25.models.features import feature_matrix, future_features, training_set

VALIDATION_FRACTION = 0.1


class XGBoostModel:
    def __init__(self, leads: tuple[int, ...] = HORIZONS,
                 covariates: tuple[str, ...] = COVARIATES,
                 future_covariates: tuple[str, ...] = (), name: str = "xgboost",
                 quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
                 n_estimators: int = 500, learning_rate: float = 0.05, max_depth: int = 6,
                 early_stopping_rounds: int = 30, random_state: int = 0):
        self.name = name
        self._leads = leads
        self._covariates = covariates
        self._future = future_covariates
        self._quantiles = quantile_levels
        self._params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                            max_depth=max_depth, early_stopping_rounds=early_stopping_rounds,
                            random_state=random_state, tree_method="hist",
                            objective="reg:quantileerror",
                            quantile_alpha=np.array(quantile_levels))
        self._models: dict[int, XGBRegressor] = {}

    def fit(self, train: pd.DataFrame) -> "XGBoostModel":
        """Train on a chronological frame; the last 10% is held out for early stopping."""
        models = {}
        for lead in self._leads:
            X, y = training_set(train, lead, self._covariates, self._future)
            split = int(len(X) * (1 - VALIDATION_FRACTION))
            model = XGBRegressor(**self._params)
            model.fit(X.iloc[:split], y.iloc[:split],
                      eval_set=[(X.iloc[split:], y.iloc[split:])], verbose=False)
            models[lead] = model
        self._models = models
        return self

    def predict(self, inputs: list[ForecastInput], horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        """(n, quantile, horizon); leads without a trained model are NaN."""
        if not self._models:
            raise RuntimeError("call fit() before predict()")
        if tuple(quantile_levels) != tuple(self._quantiles):
            raise ValueError("quantile_levels must match those used in fit()")
        if max(self._models) > horizon:
            raise ValueError(f"trained lead {max(self._models)} exceeds horizon {horizon}")
        X = feature_matrix([x.context for x in inputs], self._covariates)
        out = np.full((len(inputs), len(quantile_levels), horizon), np.nan)
        for lead, model in self._models.items():
            X_lead = X if not self._future else X.join(
                future_features([x.future for x in inputs], lead, self._future, X.index))
            # sort to remove quantile crossing
            out[:, :, lead - 1] = np.sort(model.predict(X_lead).reshape(len(inputs), -1), axis=1)
        return out
