import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from pm25.evaluation.figures import (
    MODEL_ORDER,
    drift_counts_by_month,
    plot_monthly_with_drift,
    plot_overall_mae,
)


def _overall():
    rows = [{"model": m, "lead": lead, "mae": i + lead / 10}
            for i, m in enumerate(MODEL_ORDER) for lead in (1, 6, 24)]
    return pd.DataFrame(rows)


def _monthly():
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    return pd.DataFrame([{"model": m, "lead": 24, "month": mo, "mae": 5.0}
                         for m in MODEL_ORDER for mo in months])


def _drift():
    times = pd.to_datetime(["2025-01-05", "2025-01-20", "2025-03-01"]).tz_localize("Asia/Taipei")
    return pd.DataFrame({"station": "古亭", "model": "xgboost", "lead": 24, "time": times,
                         "mean_error_after": 1.0})


def test_drift_counts_by_month_fills_missing_months():
    counts = drift_counts_by_month(_drift(), lead=24, months=[f"2025-{m:02d}" for m in (1, 2, 3)])
    assert counts.to_dict() == {"2025-01": 2, "2025-02": 0, "2025-03": 1}


def test_plot_overall_mae_writes_png(tmp_path):
    path = plot_overall_mae(_overall(), tmp_path / "overall.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_monthly_with_drift_writes_png(tmp_path):
    path = plot_monthly_with_drift(_monthly(), _drift(), lead=24, path=tmp_path / "m.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_overall_mae_rejects_unknown_model(tmp_path):
    bad = pd.concat([_overall(), pd.DataFrame([{"model": "mystery", "lead": 1, "mae": 1.0}])])
    with pytest.raises(ValueError, match="mystery"):
        plot_overall_mae(bad, tmp_path / "x.png")


def _series_preds(model, station="古亭", n=24 * 10, peak_at=None, lead=24):
    import numpy as np

    origins = pd.date_range("2025-01-01", periods=n, freq="h", tz="Asia/Taipei")
    y = np.full(n, 10.0)
    if peak_at is not None:
        y[peak_at:peak_at + 24] = 80.0
    return pd.DataFrame({"station": station, "model": model, "origin": origins, "lead": lead,
                         "target_time": origins + pd.Timedelta(hours=lead), "y": y,
                         "q10": y - 3, "q50": y, "q90": y + 3})


def test_peak_episode_finds_station_and_window():
    from pm25.evaluation.figures import peak_episode

    preds = pd.concat([_series_preds("xgboost"),
                       _series_preds("xgboost", station="小港", peak_at=24 * 5)])
    station, start, end = peak_episode(preds, model="xgboost", lead=24, window_hours=24)
    assert station == "小港"
    peak = pd.Timestamp("2025-01-06 00:00", tz="Asia/Taipei") + pd.Timedelta(hours=24)
    assert start < peak < end


def test_plot_episode_writes_png(tmp_path):
    from pm25.evaluation.figures import plot_episode

    preds = pd.concat([_series_preds(m) for m in ("xgboost", "chronos2_uni")])
    start, end = preds["target_time"].min(), preds["target_time"].max()
    path = plot_episode(preds, "古亭", start, end, lead=24,
                        models=("xgboost", "chronos2_uni"), path=tmp_path / "e.png")
    assert path.exists() and path.stat().st_size > 0
