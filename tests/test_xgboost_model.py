import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.rolling import build_inputs
from pm25.models.xgboost_model import XGBoostModel


@pytest.fixture
def frame(rng):
    idx = pd.date_range("2023-01-01", periods=24 * 120, freq="h", tz="Asia/Taipei")
    daily = 10 + 5 * np.sin(2 * np.pi * idx.hour / 24)
    return pd.DataFrame({
        "pm25": daily + rng.normal(0, 0.5, len(idx)),
        "wind_speed": rng.uniform(0, 3, len(idx)),
    }, index=idx)


def test_fit_predict_shapes_and_only_trained_leads(frame):
    model = XGBoostModel(leads=(1, 6), covariates=("wind_speed",), n_estimators=30)
    model.fit(frame[frame.index < "2023-04-01"])
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0,),
                         start="2023-04-01")
    q = model.predict(batch.inputs, horizon=6, quantile_levels=(0.1, 0.5, 0.9))
    assert q.shape == (len(batch.inputs), 3, 6)
    assert not np.isnan(q[:, :, [0, 5]]).any()
    assert np.isnan(q[:, :, 1:5]).all()  # leads that were not trained
    assert np.all(np.diff(q[:, :, 0], axis=1) >= 0)  # quantiles are non-crossing


def test_learns_daily_cycle_better_than_mean(frame):
    model = XGBoostModel(leads=(6,), covariates=(), n_estimators=100)
    model.fit(frame[frame.index < "2023-04-01"])
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0, 9),
                         start="2023-04-01")
    q = model.predict(batch.inputs, horizon=6)
    err = np.abs(q[:, 1, 5] - batch.targets[:, 5]).mean()
    baseline = np.abs(batch.targets[:, 5] - batch.targets[:, 5].mean()).mean()
    assert err < baseline / 2


def test_predict_before_fit_raises(frame):
    with pytest.raises(RuntimeError, match="fit"):
        XGBoostModel(leads=(1,)).predict([], horizon=1)


def test_lead_beyond_horizon_raises(frame):
    model = XGBoostModel(leads=(24,), covariates=(), n_estimators=5)
    model.fit(frame[frame.index < "2023-02-01"])
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0,),
                         start="2023-04-01")
    with pytest.raises(ValueError, match="horizon"):
        model.predict(batch.inputs, horizon=6)


def test_quantile_levels_must_match_fit(frame):
    model = XGBoostModel(leads=(1,), covariates=(), n_estimators=5)
    model.fit(frame[frame.index < "2023-02-01"])
    batch = build_inputs(frame, context_length=48, horizon=1, origin_hours=(0,),
                         start="2023-04-01")
    with pytest.raises(ValueError, match="quantile"):
        model.predict(batch.inputs, horizon=1, quantile_levels=(0.5,))


def test_experiment_registers_xgboost_family():
    from pm25.experiment import make_models
    assert [m.name for m in make_models("xgboost")] == ["xgboost"]


def test_future_covariates_improve_when_target_depends_on_future(rng):
    idx = pd.date_range("2023-01-01", periods=24 * 120, freq="h", tz="Asia/Taipei")
    fc = rng.uniform(0, 5, len(idx))
    pm = 20 - 3 * fc + rng.normal(0, 0.3, len(idx))  # concentration driven by wind at t
    frame = pd.DataFrame({"pm25": pm, "fc_wind_speed": fc}, index=idx)
    train, test_start = frame[frame.index < "2023-04-01"], "2023-04-01"
    batch = build_inputs(frame, context_length=48, horizon=6, origin_hours=(0, 12),
                         start=test_start)
    plain = XGBoostModel(leads=(6,), covariates=(), n_estimators=100).fit(train)
    nwp = XGBoostModel(leads=(6,), covariates=(), future_covariates=("fc_wind_speed",),
                       n_estimators=100, name="xgboost_nwp").fit(train)
    err = lambda m: np.abs(m.predict(batch.inputs, 6)[:, 1, 5] - batch.targets[:, 5]).mean()  # noqa: E731
    assert nwp.name == "xgboost_nwp"
    assert err(nwp) < err(plain) / 2


def test_make_models_nwp_xgboost():
    from pm25.experiment import make_models

    assert [m.name for m in make_models("nwp_xgboost")] == ["xgboost_nwp"]
