"""Zero-shot Chronos-Bolt and Chronos-2 wrappers (no fine-tuning).

Pipelines are loaded lazily on `default_device()` unless one is injected (tests inject mocks).
"""

import numpy as np

from pm25.config import QUANTILE_LEVELS, TARGET
from pm25.evaluation.rolling import ForecastInput
from pm25.models.device import default_device

BATCH_SIZE = 64
# Chronos-Bolt has no internal batching; one call with every origin exhausts GPU memory.
BOLT_BATCH_SIZE = 256


def _load(model_id: str):  # pragma: no cover - downloads model weights
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(model_id, device_map=default_device())


class _LazyPipeline:
    def __init__(self, name: str, model_id: str | None, pipeline):
        if pipeline is None and model_id is None:
            raise ValueError("either model_id or pipeline is required")
        self.name = name
        self._model_id = model_id
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:  # pragma: no cover
            self._pipeline = _load(self._model_id)
        return self._pipeline


class ChronosBoltModel(_LazyPipeline):
    """Univariate Chronos-Bolt (e.g. amazon/chronos-bolt-small)."""

    def __init__(self, name: str, model_id: str | None = None, pipeline=None):
        super().__init__(name, model_id, pipeline)

    def predict(self, inputs: list[ForecastInput], horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        import torch

        context = torch.tensor(np.stack([x.context[TARGET].to_numpy() for x in inputs]),
                               dtype=torch.float32)
        batches = []
        for start in range(0, len(context), BOLT_BATCH_SIZE):
            quantiles, _ = self.pipeline.predict_quantiles(
                context[start:start + BOLT_BATCH_SIZE], prediction_length=horizon,
                quantile_levels=list(quantile_levels))
            batches.append(np.asarray(quantiles.cpu(), dtype=float))
        # (batch, horizon, quantile) -> (batch, quantile, horizon)
        return np.concatenate(batches).transpose(0, 2, 1)


class Chronos2Model(_LazyPipeline):
    """Chronos-2 with optional past covariates and known-future (oracle) covariates."""

    def __init__(self, name: str, model_id: str | None = None, pipeline=None,
                 past_covariates: tuple[str, ...] = (),
                 future_covariates: tuple[str, ...] = ()):
        if not set(future_covariates) <= set(past_covariates):
            raise ValueError("future_covariates must be a subset of past_covariates")
        super().__init__(name, model_id, pipeline)
        self._past = past_covariates
        self._future = future_covariates

    def _to_dict(self, x: ForecastInput) -> dict:
        item = {"target": x.context[TARGET].to_numpy(dtype=float)}
        if self._past:
            item["past_covariates"] = {c: x.context[c].to_numpy(dtype=float) for c in self._past}
        if self._future:
            item["future_covariates"] = {c: x.future[c].to_numpy(dtype=float)
                                         for c in self._future}
        return item

    def predict(self, inputs: list[ForecastInput], horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        quantiles, _ = self.pipeline.predict_quantiles(
            [self._to_dict(x) for x in inputs], prediction_length=horizon,
            quantile_levels=list(quantile_levels), batch_size=BATCH_SIZE)
        # each item: (n_variates=1, horizon, quantile) -> (quantile, horizon)
        return np.stack([np.asarray(q.cpu(), dtype=float)[0].T for q in quantiles])
