"""Aggregate saved predictions into metric tables, monthly errors and ADWIN drift points.

Run `uv run python -m pm25.evaluation.report` after the experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    ADWIN_DELTA,
    AUTUMN_WINTER_MONTHS,
    PM25_AQI100_THRESHOLD,
    PREDICTIONS_DIR,
    STATIONS,
    TABLES_DIR,
)
from pm25.evaluation.metrics import exceedance_scores, interval_coverage, mae, rmse

REGION_OF = {station: region for region, group in STATIONS.items() for station in group}


def load_predictions(directory: Path) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no prediction files in {directory}")
    return pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)


def add_context_columns(preds: pd.DataFrame,
                        threshold: float = PM25_AQI100_THRESHOLD) -> pd.DataFrame:
    """Adds region, season (of the target hour) and episode (observed >= threshold)."""
    unknown = set(preds["station"]) - set(REGION_OF)
    if unknown:
        raise ValueError(f"no region configured for stations {sorted(unknown)}")
    autumn_winter = preds["target_time"].dt.month.isin(AUTUMN_WINTER_MONTHS)
    return preds.assign(
        region=preds["station"].map(REGION_OF),
        season=np.where(autumn_winter, "autumn_winter", "spring_summer"),
        episode=preds["y"] >= threshold,
    )


def _scores(group: pd.DataFrame, threshold: float) -> pd.Series:
    exceed = exceedance_scores(group["y"], group["q50"], threshold)
    return pd.Series({
        "mae": mae(group["y"], group["q50"]),
        "rmse": rmse(group["y"], group["q50"]),
        "coverage80": interval_coverage(group["y"], group["q10"], group["q90"]),
        "precision": exceed["precision"], "recall": exceed["recall"], "f1": exceed["f1"],
        "n": len(group),
    })


def score_table(preds: pd.DataFrame, by: tuple[str, ...] = ("model", "lead"),
                threshold: float = PM25_AQI100_THRESHOLD) -> pd.DataFrame:
    return (preds.groupby(list(by))[["y", "q10", "q50", "q90"]]
            .apply(_scores, threshold=threshold).reset_index())


def monthly_errors(preds: pd.DataFrame) -> pd.DataFrame:
    month = preds["target_time"].dt.tz_localize(None).dt.to_period("M").rename("month")
    abs_err = (preds["q50"] - preds["y"]).abs().rename("mae")
    return abs_err.groupby([preds["model"], preds["lead"], month]).mean().reset_index()


def detect_drift(preds: pd.DataFrame, delta: float = ADWIN_DELTA) -> pd.DataFrame:
    """ADWIN change points on each (station, model, lead) absolute-error stream."""
    from river.drift import ADWIN

    rows = []
    for (station, model, lead), group in preds.groupby(["station", "model", "lead"]):
        ordered = group.sort_values("target_time")
        detector = ADWIN(delta=delta)
        errors = (ordered["q50"] - ordered["y"]).abs().to_numpy()
        for time, error in zip(ordered["target_time"], errors):
            detector.update(float(error))
            if detector.drift_detected:
                rows.append({"station": station, "model": model, "lead": lead, "time": time,
                             "mean_error_after": detector.estimation})
    return pd.DataFrame(rows, columns=["station", "model", "lead", "time", "mean_error_after"])


def main() -> None:  # pragma: no cover - file IO entry point
    out = Path(TABLES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    preds = add_context_columns(load_predictions(Path(PREDICTIONS_DIR)))
    tables = {
        "overall": score_table(preds),
        "by_region": score_table(preds, by=("region", "model", "lead")),
        "by_season": score_table(preds, by=("season", "model", "lead")),
        "episodes": score_table(preds[preds["episode"]], by=("season", "model", "lead")),
        "monthly": monthly_errors(preds),
        "drift_points": detect_drift(preds),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    print(tables["overall"].round(3).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
