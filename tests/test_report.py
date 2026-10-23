import numpy as np
import pandas as pd
import pytest

from pm25.data.io import write_predictions
from pm25.evaluation.report import (
    add_context_columns,
    detect_drift,
    load_predictions,
    monthly_errors,
    score_table,
)


def _preds(model="m", station="古亭", n=48, offset=0.0, y=None):
    origins = pd.date_range("2025-01-01", periods=n, freq="h", tz="Asia/Taipei")
    y = np.full(n, 20.0) if y is None else y
    return pd.DataFrame({
        "station": station, "model": model, "origin": origins, "lead": 1,
        "target_time": origins + pd.Timedelta(hours=1), "y": y,
        "q10": y + offset - 5, "q50": y + offset, "q90": y + offset + 5,
    })


def test_load_predictions_concatenates(tmp_path):
    write_predictions(_preds("a"), tmp_path / "古亭__a.csv.gz")
    write_predictions(_preds("b"), tmp_path / "古亭__b.csv.gz")
    out = load_predictions(tmp_path)
    assert sorted(out["model"].unique()) == ["a", "b"]


def test_load_predictions_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path)


def test_add_context_columns_region_season_episode():
    out = add_context_columns(pd.concat([_preds(station="小港"),
                                         _preds(station="古亭", y=np.full(48, 40.0))]))
    assert set(out.loc[out["station"] == "小港", "region"]) == {"kaoping"}
    assert set(out["season"]) == {"autumn_winter"}
    assert out.loc[out["station"] == "古亭", "episode"].all()
    assert not out.loc[out["station"] == "小港", "episode"].any()


def test_add_context_columns_unknown_station_raises():
    with pytest.raises(ValueError, match="region"):
        add_context_columns(_preds(station="不存在"))


def test_score_table_metrics():
    preds = pd.concat([_preds("good", offset=1.0), _preds("bad", offset=-4.0)])
    table = score_table(preds, by=("model", "lead")).set_index("model")
    assert table.loc["good", "mae"] == pytest.approx(1.0)
    assert table.loc["bad", "rmse"] == pytest.approx(4.0)
    assert table.loc["good", "coverage80"] == 1.0
    assert table.loc["good", "n"] == 48
    assert {"precision", "recall", "f1"} <= set(table.columns)


def test_monthly_errors():
    preds = _preds(offset=2.0, n=24 * 40)
    out = monthly_errors(preds)
    assert list(out["month"].astype(str)) == ["2025-01", "2025-02"]
    assert np.allclose(out["mae"], 2.0)


def test_detect_drift_finds_error_jump():
    n = 24 * 60
    y = np.full(n, 20.0)
    preds = _preds(n=n, y=y)
    noise = np.random.default_rng(0).normal(0, 0.3, n)
    jump = np.where(np.arange(n) >= n // 2, 8.0, 0.0)
    preds = preds.assign(q50=y + noise + jump)
    drift = detect_drift(preds)
    assert len(drift) >= 1
    first = drift["time"].min()
    midpoint = preds["target_time"].iloc[n // 2]
    assert midpoint <= first <= midpoint + pd.Timedelta(days=5)
    assert set(drift.columns) >= {"station", "model", "lead", "time"}


def test_detect_drift_stable_stream_has_no_changes():
    n = 24 * 30
    preds = _preds(n=n)
    preds = preds.assign(q50=preds["y"] + np.random.default_rng(1).normal(0, 0.3, n))
    assert detect_drift(preds).empty


def test_score_table_q90_alert_catches_underpredicted_peaks():
    y = np.where(np.arange(48) % 2 == 0, 40.0, 10.0)
    preds = _preds(y=y).assign(q50=20.0, q90=np.where(y > 30, 35.0, 15.0))
    row = score_table(preds).iloc[0]
    assert row["recall"] == 0.0
    assert row["alert_recall"] == 1.0
    assert row["alert_precision"] == pytest.approx(1.0)
    assert {"alert_f1"} <= set(score_table(preds).columns)


def test_score_table_point_forecast_has_no_coverage():
    preds = _preds().assign(q10=lambda d: d["q50"], q90=lambda d: d["q50"])
    assert np.isnan(score_table(preds).iloc[0]["coverage80"])
