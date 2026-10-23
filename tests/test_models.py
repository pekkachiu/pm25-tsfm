from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from pm25.evaluation.rolling import ForecastInput
from pm25.models.chronos import Chronos2Model, ChronosBoltModel
from pm25.models.persistence import PersistenceModel

QL = (0.1, 0.5, 0.9)


def _inputs(n=3, context=8, horizon=4):
    idx = pd.date_range("2025-01-01", periods=context + horizon, freq="h", tz="Asia/Taipei")
    out = []
    for i in range(n):
        frame = pd.DataFrame({
            "pm25": np.arange(context + horizon, dtype=float) + i,
            "wind_speed": np.ones(context + horizon),
            "no2": np.full(context + horizon, 2.0),
        }, index=idx)
        out.append(ForecastInput(context=frame.iloc[:context],
                                 future=frame.iloc[context:].drop(columns="pm25")))
    return out


class TestPersistence:
    def test_repeats_last_value(self):
        q = PersistenceModel().predict(_inputs(), horizon=4, quantile_levels=QL)
        assert q.shape == (3, 3, 4)
        np.testing.assert_array_equal(q[:, 1, 0], [7.0, 8.0, 9.0])
        assert np.all(q[0] == 7.0)

    def test_rejects_nan_last_value(self):
        bad = _inputs(1)[0]
        ctx = bad.context.copy()
        ctx.iloc[-1, 0] = np.nan
        with pytest.raises(ValueError):
            PersistenceModel().predict([ForecastInput(ctx, bad.future)], horizon=1)


class TestChronosBolt:
    def test_passes_target_and_reorders_output(self):
        pipeline = MagicMock()
        # chronos-bolt returns (batch, horizon, quantiles)
        pipeline.predict_quantiles.return_value = (torch.arange(36.0).reshape(3, 4, 3), None)
        model = ChronosBoltModel("bolt", pipeline=pipeline)
        q = model.predict(_inputs(), horizon=4, quantile_levels=QL)
        assert q.shape == (3, 3, 4)
        assert q[0, 0, 1] == 3.0  # (sample 0, quantile 0, lead 1) == raw[0, 1, 0]
        context = pipeline.predict_quantiles.call_args.args[0]
        assert context.shape == (3, 8)
        assert pipeline.predict_quantiles.call_args.kwargs["quantile_levels"] == list(QL)


    def test_predicts_in_batches_to_bound_gpu_memory(self, monkeypatch):
        monkeypatch.setattr("pm25.models.chronos.BOLT_BATCH_SIZE", 2)
        pipeline = MagicMock()
        pipeline.predict_quantiles.side_effect = lambda ctx, **kw: (
            torch.full((len(ctx), 4, 3), float(ctx[0, -1])), None)
        q = ChronosBoltModel("bolt", pipeline=pipeline).predict(_inputs(n=5), 4, QL)
        assert pipeline.predict_quantiles.call_count == 3  # 2 + 2 + 1
        assert [len(c.args[0]) for c in pipeline.predict_quantiles.call_args_list] == [2, 2, 1]
        assert q.shape == (5, 3, 4)
        np.testing.assert_array_equal(q[:, 0, 0], [7, 7, 9, 9, 11])  # each batch's first context


class TestChronos2:
    def _pipeline(self, n=3, horizon=4):
        pipeline = MagicMock()
        # chronos-2 returns a list of (n_variates, horizon, quantiles)
        pipeline.predict_quantiles.return_value = (
            [torch.full((1, horizon, 3), float(i)) for i in range(n)], None)
        return pipeline

    def test_univariate_sends_target_only(self):
        pipeline = self._pipeline()
        q = Chronos2Model("c2", pipeline=pipeline).predict(_inputs(), 4, QL)
        assert q.shape == (3, 3, 4) and q[2, 0, 0] == 2.0
        sent = pipeline.predict_quantiles.call_args.args[0]
        assert set(sent[0]) == {"target"}
        np.testing.assert_array_equal(sent[1]["target"], np.arange(8.0) + 1)

    def test_past_covariates(self):
        pipeline = self._pipeline()
        Chronos2Model("c2", pipeline=pipeline, past_covariates=("wind_speed", "no2")) \
            .predict(_inputs(), 4, QL)
        sent = pipeline.predict_quantiles.call_args.args[0][0]
        assert set(sent["past_covariates"]) == {"wind_speed", "no2"}
        assert "future_covariates" not in sent

    def test_future_covariates_must_be_past_covariates(self):
        with pytest.raises(ValueError, match="subset"):
            Chronos2Model("c2", pipeline=MagicMock(), past_covariates=("no2",),
                          future_covariates=("wind_speed",))

    def test_oracle_future_covariates(self):
        pipeline = self._pipeline()
        Chronos2Model("c2", pipeline=pipeline, past_covariates=("wind_speed", "no2"),
                      future_covariates=("wind_speed",)).predict(_inputs(), 4, QL)
        sent = pipeline.predict_quantiles.call_args.args[0][0]
        assert set(sent["future_covariates"]) == {"wind_speed"}
        assert len(sent["future_covariates"]["wind_speed"]) == 4
        assert pipeline.predict_quantiles.call_args.kwargs["prediction_length"] == 4


def test_chronos_requires_model_id_or_pipeline():
    with pytest.raises(ValueError, match="model_id or pipeline"):
        ChronosBoltModel("bolt")
