"""Fine-tune one global Chronos-2 on all stations (phase 3).

Training uses each station's full 2021-2023 series (random windows are drawn by
Chronos2Pipeline.fit); 2024 windows select the learning rate and the best checkpoint;
2025 stays untouched for the test. For the 2024 test (phase 5) both windows move one year
earlier (config.FINETUNE_SPLITS). Run on a GPU:

    python -m pm25.models.finetune --variant chronos2_ft_nwp [--test-year 2024]
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    CONTEXT_LENGTH,
    COVARIATES,
    FINETUNE_BATCH_SIZE,
    FINETUNE_LEARNING_RATES,
    FINETUNE_STEPS,
    FINETUNE_TRAIN,
    FINETUNE_VAL_EVERY_HOURS,
    FINETUNE_VALID,
    FINETUNED_DIR,
    FORECAST_LENGTH,
    PRIMARY_NWP,
    PROCESSED_DATA_DIR,
    STATIONS,
    TARGET,
    TEST_YEAR,
    TEST_YEARS,
    WIND_FORECAST_COVARIATES,
)
from pm25.paths import finetune_split, for_year


@dataclass(frozen=True)
class FinetuneSpec:
    name: str
    past_covariates: tuple[str, ...] = ()
    future_covariates: tuple[str, ...] = ()  # known in the future (e.g. NWP wind)

    def __post_init__(self):
        if not set(self.future_covariates) <= set(self.past_covariates):
            raise ValueError("future_covariates must be a subset of past_covariates")


def _series_dict(frame: pd.DataFrame, spec: FinetuneSpec) -> dict:
    item = {"target": frame[TARGET].to_numpy(dtype=float)}
    if spec.past_covariates:
        item["past_covariates"] = {c: frame[c].to_numpy(dtype=float)
                                   for c in spec.past_covariates}
    return item


def training_series(frames: dict[str, pd.DataFrame], spec: FinetuneSpec, start: str,
                    end: str) -> list[dict]:
    """One full-length series per station between start and end (inclusive)."""
    return [_series_dict(df.loc[start:end], spec) for df in frames.values()]


def validation_windows(frames: dict[str, pd.DataFrame], spec: FinetuneSpec, start: str,
                       end: str, context_length: int = CONTEXT_LENGTH,
                       horizon: int = FORECAST_LENGTH,
                       every_hours: int = FINETUNE_VAL_EVERY_HOURS) -> list[dict]:
    """Series of context + horizon hours whose complete horizon lies in [start, end];
    Chronos-2 scores the last `horizon` hours of each (validation mode)."""
    windows = []
    for df in frames.values():
        tz = df.index.tz
        lo, hi = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
        position = {t: i for i, t in enumerate(df.index)}
        for origin in pd.date_range(lo - pd.Timedelta(hours=1), hi, freq=f"{every_hours}h"):
            i = position.get(origin)
            if i is None or i + 1 < context_length or i + horizon >= len(df):
                continue
            window = df.iloc[i + 1 - context_length:i + 1 + horizon]
            target = window[TARGET].to_numpy(dtype=float)
            if window.index[-1] > hi or np.isnan(target[-horizon:]).any():
                continue
            windows.append(_series_dict(window, spec))
    return windows


def _prepare(dicts: list[dict], spec: FinetuneSpec, horizon: int):
    from chronos.chronos2.preprocess import from_list_of_dicts

    known = list(spec.future_covariates) or None
    return from_list_of_dicts(dicts, prediction_length=horizon, known_covariates_names=known)


def _loss_recorder():
    from transformers import TrainerCallback

    class LossRecorder(TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                self.losses.append(float(metrics["eval_loss"]))

    return LossRecorder()


def finetune(base_pipeline, spec: FinetuneSpec, frames: dict[str, pd.DataFrame],
             out_dir: Path, train: tuple[str, str] = FINETUNE_TRAIN,
             valid: tuple[str, str] = FINETUNE_VALID,
             learning_rates: tuple[float, ...] = FINETUNE_LEARNING_RATES,
             num_steps: int = FINETUNE_STEPS, batch_size: int = FINETUNE_BATCH_SIZE,
             context_length: int = CONTEXT_LENGTH, horizon: int = FORECAST_LENGTH,
             val_every_hours: int = FINETUNE_VAL_EVERY_HOURS) -> dict:
    """Full fine-tuning per learning rate; keeps the run with the lowest best validation
    loss (each run already restores its best checkpoint) and saves it to out_dir/name."""
    out_dir = Path(out_dir)
    train_inputs = _prepare(training_series(frames, spec, *train), spec, horizon)
    valid_inputs = _prepare(validation_windows(frames, spec, *valid, context_length,
                                               horizon, val_every_hours), spec, horizon)
    losses, best = {}, None
    for lr in learning_rates:
        recorder = _loss_recorder()
        run_dir = out_dir / "_runs" / f"{spec.name}_lr{lr:g}"
        tuned = base_pipeline.fit(
            inputs=train_inputs, prediction_length=horizon, validation_inputs=valid_inputs,
            learning_rate=lr, callbacks=[recorder], finetune_mode="full",
            num_steps=num_steps, batch_size=batch_size, context_length=context_length,
            output_dir=run_dir)
        # the tuned model is in memory; trainer checkpoints (~1 GB per run) are not needed
        shutil.rmtree(run_dir, ignore_errors=True)
        losses[f"{lr:g}"] = min(recorder.losses) if recorder.losses else float("nan")
        if best is None or losses[f"{lr:g}"] < best[1]:
            best = (lr, losses[f"{lr:g}"], tuned)
    best[2].save_pretrained(out_dir / spec.name)
    summary = {"variant": spec.name, "best_learning_rate": best[0], "validation_loss": losses,
               "num_steps": num_steps, "batch_size": batch_size, "train": list(train),
               "valid": list(valid), "n_train_series": len(train_inputs),
               "n_valid_windows": len(valid_inputs)}
    (out_dir / f"{spec.name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def _variants() -> dict[str, FinetuneSpec]:
    specs = [
        FinetuneSpec("chronos2_ft_uni"),
        FinetuneSpec("chronos2_ft_cov", past_covariates=COVARIATES),
        FinetuneSpec("chronos2_ft_nwp", past_covariates=COVARIATES + WIND_FORECAST_COVARIATES,
                     future_covariates=WIND_FORECAST_COVARIATES),
    ]
    return {s.name: s for s in specs}


FINETUNE_VARIANTS = _variants()


def main() -> None:  # pragma: no cover - GPU entry point
    from pm25.data.io import read_station
    from pm25.data.wind_forecast import attach_wind_forecast, forecast_dir
    from pm25.models.chronos import _load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(FINETUNE_VARIANTS), nargs="+",
                        default=sorted(FINETUNE_VARIANTS))
    parser.add_argument("--num-steps", type=int, default=FINETUNE_STEPS)
    parser.add_argument("--test-year", type=int, choices=TEST_YEARS, default=TEST_YEAR)
    args = parser.parse_args()
    out_dir = Path(for_year(FINETUNED_DIR, args.test_year))
    train, valid = finetune_split(args.test_year)
    frames = {}
    for station in (s for group in STATIONS.values() for s in group):
        df = read_station(Path(PROCESSED_DATA_DIR), station).drop(columns="station")
        frames[station] = attach_wind_forecast(
            df, read_station(forecast_dir(PRIMARY_NWP), station))
    for name in args.variant:
        if (out_dir / name).is_dir():  # resume after a disconnected runtime
            print(f"{name}: checkpoint exists, skipped", flush=True)
            continue
        summary = finetune(_load("amazon/chronos-2"), FINETUNE_VARIANTS[name], frames,
                           out_dir, train=train, valid=valid, num_steps=args.num_steps)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
