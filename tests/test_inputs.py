import numpy as np
import pandas as pd
import pytest

from pm25.evaluation.rolling import build_inputs


@pytest.fixture
def frame():
    idx = pd.date_range("2025-01-01", periods=24 * 6, freq="h", tz="Asia/Taipei")
    n = len(idx)
    return pd.DataFrame({
        "pm25": np.arange(n, dtype=float),
        "wind_speed": np.arange(n, dtype=float) * 10,
        "no2": np.full(n, np.nan),  # covariate gaps do not invalidate windows
    }, index=idx)


def test_build_inputs_slices_context_future_and_targets(frame):
    batch = build_inputs(frame, context_length=24, horizon=6, origin_hours=(0,))
    first = batch.inputs[0]
    assert batch.origins[0] == pd.Timestamp("2025-01-02", tz="Asia/Taipei")
    assert len(first.context) == 24 and first.context.index[-1] == batch.origins[0]
    assert len(first.future) == 6
    assert first.future.index[0] == batch.origins[0] + pd.Timedelta(hours=1)
    assert "pm25" not in first.future.columns
    np.testing.assert_array_equal(batch.targets[0], first.context["pm25"].iloc[-1] + np.arange(1, 7))
    assert batch.targets.shape == (len(batch.inputs), 6)


def test_build_inputs_drops_windows_with_missing_target(frame):
    gappy = frame.assign(pm25=frame["pm25"].where(frame.index.day != 3))
    batch = build_inputs(gappy, context_length=24, horizon=6, origin_hours=(0,))
    assert pd.Timestamp("2025-01-03", tz="Asia/Taipei") not in batch.origins
    assert len(batch.inputs) == len(batch.origins) == len(batch.targets)
