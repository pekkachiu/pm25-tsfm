"""Rolling-origin experiment runner.

Each model family runs in its own process because torch and xgboost cannot share one
on macOS (see CLAUDE.md):

    uv run python -m pm25.experiment --family <family> [--stations ...]

nwp_* families add day-ahead NWP wind (data/wind_forecast/<model>/) as future covariates;
spatial_* families also add other stations' PM2.5 as past covariates (phase 4).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    CONTEXT_LENGTH,
    COVARIATES,
    FINETUNED_DIR,
    FORECAST_LENGTH,
    HORIZONS,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    QUANTILE_LEVELS,
    STATIONS,
    TEST_YEAR,
    WIND_COVARIATES,
    PRIMARY_NWP,
    WIND_FORECAST_COVARIATES,
)
from pm25.data.spatial import attach_neighbours, spatial_columns
from pm25.data.io import read_station, write_predictions
from pm25.data.split import split_by_year
from pm25.data.wind_forecast import attach_wind_forecast, forecast_dir
from pm25.evaluation.rolling import build_inputs

FAMILIES = ("baseline", "xgboost", "lstm", "chronos", "nwp_xgboost", "nwp_torch",
            "nwp_bestmatch", "finetuned", "spatial_xgboost", "spatial_chronos")
# Families that need NWP wind, and the NWP model they read. best_match is archived only
# from 2024, so it is limited to the zero-shot Chronos-2 sensitivity run.
FAMILY_NWP = {"nwp_bestmatch": "best_match", "nwp_xgboost": PRIMARY_NWP,
              "nwp_torch": PRIMARY_NWP, "finetuned": PRIMARY_NWP,
              "spatial_xgboost": PRIMARY_NWP, "spatial_chronos": PRIMARY_NWP}
SPATIAL_FAMILIES = ("spatial_xgboost", "spatial_chronos")
NWP_INPUTS = dict(past_covariates=COVARIATES + WIND_FORECAST_COVARIATES,
                  future_covariates=WIND_FORECAST_COVARIATES)
NEAREST = spatial_columns("nearest")


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
    if family == "nwp_xgboost":
        from pm25.models.xgboost_model import XGBoostModel
        return [XGBoostModel(future_covariates=WIND_FORECAST_COVARIATES, name="xgboost_nwp")]
    if family == "nwp_torch":
        from pm25.models.chronos import Chronos2Model
        from pm25.models.lstm_model import LSTMModel
        return [LSTMModel(future_covariates=WIND_FORECAST_COVARIATES, name="lstm_nwp"),
                Chronos2Model("chronos2_wind_nwp", "amazon/chronos-2", **NWP_INPUTS)]
    if family == "nwp_bestmatch":
        from pm25.models.chronos import Chronos2Model
        return [Chronos2Model("chronos2_wind_bestmatch", "amazon/chronos-2", **NWP_INPUTS)]
    if family == "finetuned":
        from pm25.models.chronos import Chronos2Model
        from pm25.models.finetune import FINETUNE_VARIANTS
        # only variants whose checkpoint exists (see pm25.models.finetune)
        return [Chronos2Model(name, str(Path(FINETUNED_DIR) / name),
                              past_covariates=spec.past_covariates,
                              future_covariates=spec.future_covariates)
                for name, spec in FINETUNE_VARIANTS.items()
                if (Path(FINETUNED_DIR) / name).is_dir()]
    if family == "spatial_xgboost":
        from pm25.models.xgboost_model import XGBoostModel
        return [XGBoostModel(covariates=COVARIATES + NEAREST, name="xgboost_spatial"),
                XGBoostModel(covariates=COVARIATES + spatial_columns("all"),
                             name="xgboost_spatial_all"),
                XGBoostModel(covariates=COVARIATES + NEAREST,
                             future_covariates=WIND_FORECAST_COVARIATES,
                             name="xgboost_spatial_nwp")]
    if family == "spatial_chronos":
        from pm25.models.chronos import Chronos2Model
        scoped = [Chronos2Model(name, "amazon/chronos-2",
                                past_covariates=COVARIATES + spatial_columns(scope))
                  for name, scope in (("chronos2_spatial", "nearest"),
                                      ("chronos2_spatial_region", "region"),
                                      ("chronos2_spatial_all", "all"))]
        return [*scoped, Chronos2Model(
            "chronos2_spatial_nwp", "amazon/chronos-2",
            past_covariates=COVARIATES + NEAREST + WIND_FORECAST_COVARIATES,
            future_covariates=WIND_FORECAST_COVARIATES)]
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
        path = out_dir / f"{station}__{model.name}.csv.gz"
        paths.append(path)
        if path.exists():
            continue
        started = time.time()
        if hasattr(model, "fit"):
            model.fit(train)
        quantiles = model.predict(batch.inputs, FORECAST_LENGTH, QUANTILE_LEVELS)
        write_predictions(prediction_frame(station, model.name, batch.origins,
                                           batch.targets, quantiles), path)
        print(f"{station} {model.name}: {len(batch.origins)} origins, "
              f"{time.time() - started:.0f}s", flush=True)
    return paths


def load_station(station: str, family: str, processed_dir: Path = Path(PROCESSED_DATA_DIR),
                 nwp_root: Path | None = None) -> pd.DataFrame:
    """Processed station frame plus the inputs the family needs (NWP wind, neighbours)."""
    def read(name: str) -> pd.DataFrame:
        return read_station(processed_dir, name).drop(columns="station")

    df = read(station)
    if family in FAMILY_NWP:
        nwp_dir = (Path(nwp_root) / FAMILY_NWP[family] if nwp_root is not None
                   else forecast_dir(FAMILY_NWP[family]))
        df = attach_wind_forecast(df, read_station(nwp_dir, station))
    if family in SPATIAL_FAMILIES:
        others = {s: read(s) for group in STATIONS.values() for s in group if s != station}
        df = attach_neighbours(df, station, others)
    return df


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--stations", nargs="*",
                        default=[s for group in STATIONS.values() for s in group])
    args = parser.parse_args()
    for station in args.stations:
        run_station(station, load_station(station, args.family), make_models(args.family),
                    Path(PREDICTIONS_DIR))


if __name__ == "__main__":  # pragma: no cover
    main()
