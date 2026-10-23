"""Significance tests, per-station comparisons and drift recovery times.

Run `uv run python -m pm25.evaluation.analysis` after `pm25.evaluation.report`.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import DM_HAC_LAG, DM_PAIRS, HORIZONS, PREDICTIONS_DIR, TABLES_DIR

SIGNIFICANCE_LEVEL = 0.05


def _abs_errors(preds: pd.DataFrame, lead: int) -> pd.DataFrame:
    """(station, target_time) x model table of absolute errors for one lead."""
    sub = preds[preds["lead"] == lead]
    return (sub.assign(abs_err=(sub["q50"] - sub["y"]).abs())
            .pivot_table(index=["station", "target_time"], columns="model", values="abs_err"))


def loss_differential(preds: pd.DataFrame, model: str, baseline: str, lead: int) -> pd.Series:
    """|e_model| - |e_baseline| averaged over stations at each target time (positive = model
    worse). Only hours both models forecast are kept."""
    errors = _abs_errors(preds[preds["model"].isin((model, baseline))], lead)
    diff = (errors[model] - errors[baseline]).dropna()
    return diff.groupby(level="target_time").mean().sort_index()


def diebold_mariano(d: pd.Series, lag: int = DM_HAC_LAG) -> dict:
    """Two-sided DM test of E[d] = 0 with a Newey-West (Bartlett) long-run variance."""
    values = np.asarray(d, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    centered = values - values.mean()
    variance = centered @ centered / n
    for k in range(1, min(lag, n - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * (centered[k:] @ centered[:-k]) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = values.mean() / np.sqrt(variance / n)
    p_value = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else (
        0.0 if values.mean() != 0 else 1.0)
    return {"mean_diff": float(values.mean()), "dm_stat": float(stat),
            "p_value": float(p_value), "n": n}


def _available(preds: pd.DataFrame, pairs) -> list[tuple[str, str]]:
    models = set(preds["model"])
    return [(a, b) for a, b in pairs if a in models and b in models]


def pairwise_tests(preds: pd.DataFrame, pairs=DM_PAIRS, leads=HORIZONS,
                   lag: int = DM_HAC_LAG) -> pd.DataFrame:
    """Station-pooled DM test per (model, baseline, lead); pct_change is relative to baseline
    MAE (negative = model better)."""
    rows = []
    for model, baseline in _available(preds, pairs):
        for lead in leads:
            d = loss_differential(preds, model, baseline, lead)
            base_mae = _abs_errors(preds[preds["model"] == baseline], lead)[baseline].mean()
            rows.append({"model": model, "baseline": baseline, "lead": lead,
                         **diebold_mariano(d, lag), "pct_change": 100 * d.mean() / base_mae})
    return pd.DataFrame(rows, columns=["model", "baseline", "lead", "mean_diff", "dm_stat",
                                       "p_value", "n", "pct_change"])


def station_wins(preds: pd.DataFrame, pairs=DM_PAIRS, leads=HORIZONS,
                 lag: int = DM_HAC_LAG) -> pd.DataFrame:
    """Per pair and lead: stations where the model has lower MAE, and how many of those
    (or of the losses) are significant under a per-station DM test."""
    rows = []
    for model, baseline in _available(preds, pairs):
        for lead in leads:
            wins = sig_wins = sig_losses = 0
            stations = sorted(preds["station"].unique())
            for station in stations:
                d = loss_differential(preds[preds["station"] == station], model, baseline, lead)
                test = diebold_mariano(d, lag)
                better = test["mean_diff"] < 0
                significant = test["p_value"] < SIGNIFICANCE_LEVEL
                wins += better
                sig_wins += better and significant
                sig_losses += (not better) and significant
            rows.append({"model": model, "baseline": baseline, "lead": lead, "wins": wins,
                         "stations": len(stations), "sig_wins": sig_wins,
                         "sig_losses": sig_losses})
    return pd.DataFrame(rows)


def recovery_times(drift: pd.DataFrame) -> pd.DataFrame:
    """Hours from each ADWIN change that raised the error level to the next change that
    lowered it, per (station, model, lead) stream. Unrecovered rises are dropped."""
    rows = []
    for (station, model, lead), stream in drift.groupby(["station", "model", "lead"]):
        ordered = stream.sort_values("time")
        times = list(pd.to_datetime(ordered["time"]))
        step = np.diff(ordered["mean_error_after"].to_numpy())
        for i, rise in enumerate(step, start=1):
            if rise <= 0:
                continue
            falls = [j for j in range(i + 1, len(times)) if step[j - 1] < 0]
            if falls:
                hours = (times[falls[0]] - times[i]).total_seconds() / 3600
                rows.append({"station": station, "model": model, "lead": lead,
                             "start": times[i], "hours": hours})
    return pd.DataFrame(rows, columns=["station", "model", "lead", "start", "hours"])


def station_summary(df: pd.DataFrame, year: int, threshold: float) -> dict:
    """Mean PM2.5, exceedance hours and missing fraction of one station in one year."""
    pm25 = df.loc[df.index.year == year, "pm25"]
    return {"mean": pm25.mean(), "exceed_hours": int((pm25 >= threshold).sum()),
            "missing": pm25.isna().mean()}


def main() -> None:  # pragma: no cover - file IO entry point
    from pm25.evaluation.report import load_predictions, score_table

    out = Path(TABLES_DIR)
    preds = load_predictions(Path(PREDICTIONS_DIR))
    drift = pd.read_csv(out / "drift_points.csv")
    drift["time"] = pd.to_datetime(drift["time"], utc=True)
    tables = {
        "dm_tests": pairwise_tests(preds),
        "station_wins": station_wins(preds),
        "by_station": score_table(preds, by=("station", "model", "lead")),
        "recovery": (recovery_times(drift).groupby(["model", "lead"])["hours"]
                     .agg(["median", "mean", "count"]).reset_index()),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    print(tables["dm_tests"].round(4).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
