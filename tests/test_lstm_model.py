import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.rolling import build_inputs
from pm25.models.lstm_model import LSTMModel, pinball_loss


@pytest.fixture
def frame(rng):
    idx = pd.date_range("2023-01-01", periods=24 * 90, freq="h", tz="Asia/Taipei")
    daily = 10 + 5 * np.sin(2 * np.pi * idx.hour / 24)
    df = pd.DataFrame({
        "pm25": daily + rng.normal(0, 0.5, len(idx)),
        "wind_speed": rng.uniform(0, 3, len(idx)),
    }, index=idx)
    df.iloc[100:103, 1] = np.nan  # covariate gaps must not break training
    return df


def _fit(frame, **kw):
    params = dict(leads=(1, 6), covariates=("wind_speed",), hidden_size=16,
                  max_epochs=15, patience=5, seed=0)
    return LSTMModel(**{**params, **kw}).fit(frame[frame.index < "2023-03-01"])


def test_pinball_loss_matches_definition():
    import torch
    y = torch.tensor([[1.0]])
    pred = torch.tensor([[[0.0, 1.0, 3.0]]])  # (batch, lead, quantile)
    loss = pinball_loss(pred, y, torch.tensor([0.1, 0.5, 0.9]))
    expected = np.mean([0.1 * 1.0, 0.0, (1 - 0.9) * 2.0])
    assert loss.item() == pytest.approx(expected)


def test_predict_shapes_and_untrained_leads_are_nan(frame):
    model = _fit(frame)
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0,),
                         start="2023-03-01")
    q = model.predict(batch.inputs, horizon=6, quantile_levels=(0.1, 0.5, 0.9))
    assert q.shape == (len(batch.inputs), 3, 6)
    assert not np.isnan(q[:, :, [0, 5]]).any()
    assert np.isnan(q[:, :, 1:5]).all()
    assert np.all(np.diff(q[:, :, 0], axis=1) >= 0)


def test_learns_daily_cycle_better_than_mean(frame):
    model = _fit(frame, leads=(6,), max_epochs=40)
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0, 9),
                         start="2023-03-01")
    q = model.predict(batch.inputs, horizon=6)
    err = np.abs(q[:, 1, 5] - batch.targets[:, 5]).mean()
    baseline = np.abs(batch.targets[:, 5] - batch.targets[:, 5].mean()).mean()
    assert err < baseline / 2


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        LSTMModel(leads=(1,)).predict([], horizon=1)


def test_lead_beyond_horizon_and_quantile_mismatch_raise(frame):
    model = _fit(frame, leads=(6,), max_epochs=1)
    batch = build_inputs(frame, context_length=48, horizon=1, origin_hours=(0,),
                         start="2023-03-01")
    with pytest.raises(ValueError, match="horizon"):
        model.predict(batch.inputs, horizon=1)
    with pytest.raises(ValueError, match="quantile"):
        model.predict(batch.inputs, horizon=6, quantile_levels=(0.5,))


def test_experiment_registers_lstm_family():
    from pm25.experiment import make_models
    assert [m.name for m in make_models("lstm")] == ["lstm"]


def test_future_covariates_help_when_target_depends_on_future(rng):
    idx = pd.date_range("2023-01-01", periods=24 * 90, freq="h", tz="Asia/Taipei")
    fc = rng.uniform(0, 5, len(idx))
    frame = pd.DataFrame({"pm25": 20 - 3 * fc + rng.normal(0, 0.3, len(idx)),
                          "fc_wind_speed": fc}, index=idx)
    frame.iloc[200:203, 1] = np.nan
    train = frame[frame.index < "2023-03-01"]
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0, 12),
                         start="2023-03-01")
    params = dict(leads=(6,), covariates=(), hidden_size=16, max_epochs=80, patience=10)
    plain = LSTMModel(**params).fit(train)
    nwp = LSTMModel(**params, future_covariates=("fc_wind_speed",), name="lstm_nwp").fit(train)
    err = lambda m: np.abs(m.predict(batch.inputs, 6)[:, 1, 5] - batch.targets[:, 5]).mean()  # noqa: E731
    assert nwp.name == "lstm_nwp" and plain.name == "lstm"
    assert err(nwp) < err(plain) / 2
