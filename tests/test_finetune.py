from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pm25.models.finetune import (
    FinetuneSpec,
    finetune,
    training_series,
    validation_windows,
)


def _frame(start="2023-12-01", end="2024-02-29 23:00", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="Asia/Taipei")
    return pd.DataFrame({"pm25": rng.uniform(5, 40, len(idx)),
                         "wind_speed": rng.uniform(0, 3, len(idx)),
                         "fc_wind_speed": rng.uniform(0, 3, len(idx))}, index=idx)


SPEC_COV = FinetuneSpec("ft_cov", past_covariates=("wind_speed",))
SPEC_NWP = FinetuneSpec("ft_nwp", past_covariates=("wind_speed", "fc_wind_speed"),
                        future_covariates=("fc_wind_speed",))


def test_spec_rejects_future_not_in_past():
    with pytest.raises(ValueError, match="subset"):
        FinetuneSpec("bad", past_covariates=(), future_covariates=("fc_wind_speed",))


def test_training_series_one_dict_per_station_within_window():
    frames = {"古亭": _frame(), "小港": _frame(seed=1)}
    series = training_series(frames, SPEC_COV, "2023-12-01", "2023-12-31 23:00")
    assert len(series) == 2
    assert len(series[0]["target"]) == 31 * 24
    assert set(series[0]["past_covariates"]) == {"wind_speed"}
    assert "future_covariates" not in series[0]


def test_training_series_univariate_has_no_covariates():
    series = training_series({"古亭": _frame()}, FinetuneSpec("ft_uni"), "2023-12-01",
                             "2023-12-31 23:00")
    assert set(series[0]) == {"target"}


def test_validation_windows_have_context_plus_horizon_and_complete_targets():
    df = _frame()
    df.loc["2024-01-10 05:00", "pm25"] = np.nan
    windows = validation_windows({"古亭": df}, SPEC_NWP, "2024-01-01", "2024-01-31 23:00",
                                 context_length=48, horizon=24, every_hours=24)
    assert all(len(w["target"]) == 48 + 24 for w in windows)
    assert all(len(w["past_covariates"]["fc_wind_speed"]) == 72 for w in windows)
    assert all(not np.isnan(w["target"][-24:]).any() for w in windows)
    # 31 daily origins in January minus the one whose target hits the gap, minus the last
    # origins whose 24 h target would leave the window
    assert 25 <= len(windows) <= 30


class _FakePipeline:
    """Records fit calls; reports a validation loss that depends on the learning rate."""

    def __init__(self, losses):
        self.losses, self.calls, self.saved = losses, [], None

    def fit(self, inputs, prediction_length, validation_inputs, learning_rate, callbacks,
            **kwargs):
        (kwargs["output_dir"] / "checkpoint-100").mkdir(parents=True)
        self.calls.append({"lr": learning_rate, "n_train": len(inputs),
                           "n_val": len(validation_inputs), **kwargs})
        for loss in (self.losses[learning_rate] + 1, self.losses[learning_rate]):
            for cb in callbacks:
                cb.on_evaluate(None, None, None, metrics={"eval_loss": loss})
        return SimpleNamespace(lr=learning_rate, save_pretrained=self._save(learning_rate))

    def _save(self, lr):
        def save(path):
            self.saved = (lr, str(path))
        return save


def test_finetune_selects_lowest_validation_loss_and_saves(tmp_path):
    frames = {"古亭": _frame(), "小港": _frame(seed=1)}
    base = _FakePipeline({1e-6: 0.9, 1e-5: 0.4})
    summary = finetune(base, SPEC_NWP, frames, out_dir=tmp_path,
                       train=("2023-12-01", "2023-12-31 23:00"),
                       valid=("2024-01-01", "2024-01-31 23:00"),
                       learning_rates=(1e-6, 1e-5), num_steps=3, context_length=48,
                       horizon=24, val_every_hours=48)
    assert [c["lr"] for c in base.calls] == [1e-6, 1e-5]
    assert summary["best_learning_rate"] == 1e-5
    assert summary["validation_loss"] == {"1e-06": 0.9, "1e-05": 0.4}
    assert base.saved == (1e-5, str(tmp_path / "ft_nwp"))
    assert base.calls[0]["num_steps"] == 3 and base.calls[0]["context_length"] == 48
    assert base.calls[0]["n_train"] == 2 and base.calls[0]["n_val"] > 0
    assert (tmp_path / "ft_nwp.json").exists()
    assert not any((tmp_path / "_runs").glob("*"))  # trainer checkpoints removed
