"""Rolling-origin experiment runner.

Each model family runs in its own process because torch and xgboost cannot share one
on macOS (see CLAUDE.md):

    uv run python -m pm25.experiment --family baseline|xgboost|lstm|chronos [--stations ...]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    CONTEXT_LENGTH,
    COVARIATES,
    FORECAST_LENGTH,
    HORIZONS,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    QUANTILE_LEVELS,
    STATIONS,
    TEST_YEAR,
    WIND_COVARIATES,
)
from pm25.data.split import split_by_year
from pm25.evaluation.rolling import build_inputs

FAMILIES = ("baseline", "xgboost", "lstm", "chronos")


def make_models(family: str) -> list:
    """Fresh model instances for one family; imports stay local to keep families isolated."""
    if family == "baseline":
        from pm25.models.persistence import PersistenceModel
        return [PersistenceModel()]
    if family == "xgboost":
        from pm25.models.xgboost_model import XGBoostModel
        return [XGBoostModel()]
    if family == "lstm":
        from pm25.models.lstm_model import LSTMModel
        return [LSTMModel()]
    if family == "chronos":  # pragma: no cover - needs model weights
        from pm25.models.chronos import Chronos2Model, ChronosBoltModel
        return [
            ChronosBoltModel("chronos_bolt_small", "amazon/chronos-bolt-small"),
            ChronosBoltModel("chronos_bolt_base", "amazon/chronos-bolt-base"),
            Chronos2Model("chronos2_uni", "amazon/chronos-2"),
            Chronos2Model("chronos2_cov", "amazon/chronos-2", past_covariates=COVARIATES),
            Chronos2Model("chronos2_wind_oracle", "amazon/chronos-2",
                          past_covariates=COVARIATES, future_covariates=WIND_COVARIATES),
        ]
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


def prediction_frame(station: str, model: str, origins: list[pd.Timestamp],
                     targets: np.ndarray, quantiles: np.ndarray,
                     leads: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Long format: one row per (origin, lead) with truth and q10/q50/q90."""
    origin_index = pd.DatetimeIndex(origins)
    frames = [
        pd.DataFrame({
            "station": station, "model": model, "origin": origin_index, "lead": lead,
            "target_time": origin_index + pd.Timedelta(hours=lead),
            "y": targets[:, lead - 1],
            **{f"q{round(level * 100)}": quantiles[:, i, lead - 1]
               for i, level in enumerate(QUANTILE_LEVELS)},
        })
        for lead in leads
    ]
    return pd.concat(frames, ignore_index=True)


def run_station(station: str, df: pd.DataFrame, models: list, out_dir: Path,
                test_year: int = TEST_YEAR, context_length: int = CONTEXT_LENGTH) -> list[Path]:
    """Forecast every hourly origin of the test year; existing outputs are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, _ = split_by_year(df, test_year)
    batch = build_inputs(df, context_length, FORECAST_LENGTH,
                         start=f"{test_year}-01-01", end=f"{test_year}-12-31 23:00")
    paths = []
    for model in models:
        path = out_dir / f"{station}__{model.name}.pkl"
        paths.append(path)
        if path.exists():
            continue
        started = time.time()
        if hasattr(model, "fit"):
            model.fit(train)
        quantiles = model.predict(batch.inputs, FORECAST_LENGTH, QUANTILE_LEVELS)
        prediction_frame(station, model.name, batch.origins, batch.targets,
                         quantiles).to_pickle(path)
        print(f"{station} {model.name}: {len(batch.origins)} origins, "
              f"{time.time() - started:.0f}s", flush=True)
    return paths


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--stations", nargs="*",
                        default=[s for group in STATIONS.values() for s in group])
    args = parser.parse_args()
    for station in args.stations:
        df = pd.read_pickle(Path(PROCESSED_DATA_DIR) / f"{station}.pkl").drop(columns="station")
        run_station(station, df, make_models(args.family), Path(PREDICTIONS_DIR))


if __name__ == "__main__":  # pragma: no cover
    main()
