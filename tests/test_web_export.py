import json

import numpy as np
import pandas as pd
import pytest

from pm25.data.io import write_predictions
from pm25.web.export import (
    build_index,
    build_station,
    export_all,
    load_station_frames,
    station_file,
)

MODELS = ("a", "b")


def _preds(model, station="古亭", n=6, offset=0.0, leads=(1, 6, 24), drop=()):
    rows = []
    origins = pd.date_range("2025-01-01", periods=n, freq="h", tz="Asia/Taipei")
    for lead in leads:
        y = np.arange(n, dtype=float) + 10
        rows.append(pd.DataFrame({
            "station": station, "model": model, "origin": origins, "lead": lead,
            "target_time": origins + pd.Timedelta(hours=lead), "y": y,
            "q10": y + offset - 2.04, "q50": y + offset, "q90": y + offset + 2.06,
        }))
    out = pd.concat(rows, ignore_index=True)
    return out.drop(index=list(drop)).reset_index(drop=True)


def _frames(**kw):
    return {"a": _preds("a", **kw), "b": _preds("b", offset=1.0, **kw)}


def test_build_station_dense_hourly_arrays():
    out = build_station(_frames(), "古亭", 2025)
    assert out["station"] == "古亭" and out["year"] == 2025
    assert out["start"] == "2025-01-01T01:00:00+08:00"  # earliest target time
    # target times span 01:00 (lead 1) .. 05:00 + 24 h (lead 24) -> 29 hourly slots
    lead1 = out["leads"]["1"]
    assert len(lead1["y"]) == 29
    assert lead1["y"][:6] == [10, 11, 12, 13, 14, 15]
    assert lead1["y"][6:] == [None] * 23  # hours without a lead-1 target are null
    # lead 24 starts 23 slots later
    assert out["leads"]["24"]["y"][23:] == [10, 11, 12, 13, 14, 15]


def test_build_station_rounds_forecasts():
    out = build_station(_frames(), "古亭", 2025)
    pred = out["leads"]["1"]["models"]["b"]
    assert pred["q50"][0] == 11.0
    assert pred["q10"][0] == 9.0  # 10 + 1 - 2.04 = 8.96 -> 9.0
    assert pred["q90"][0] == 13.1  # 10 + 1 + 2.06 = 13.06 -> 13.1


def test_build_station_metrics_before_rounding():
    out = build_station(_frames(), "古亭", 2025)
    metrics = out["metrics"]
    assert metrics["a"]["24"]["mae"] == pytest.approx(0.0)
    assert metrics["b"]["24"]["mae"] == pytest.approx(1.0)
    assert metrics["b"]["1"]["coverage80"] == pytest.approx(1.0)
    assert metrics["a"]["1"]["n"] == 6


def test_build_station_rejects_misaligned_models():
    frames = {"a": _preds("a"), "b": _preds("b", drop=(0,))}
    with pytest.raises(ValueError, match="b"):
        build_station(frames, "古亭", 2025)


def test_build_station_rejects_conflicting_targets():
    b = _preds("b")
    b.loc[0, "y"] = 99.0
    with pytest.raises(ValueError, match="y"):
        build_station({"a": _preds("a"), "b": b}, "古亭", 2025)


def test_build_station_rejects_missing_values():
    a = _preds("a")
    a.loc[0, "q50"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        build_station({"a": a, "b": _preds("b")}, "古亭", 2025)


def test_build_station_rejects_unexpected_leads():
    frames = {"a": _preds("a", leads=(1, 6)), "b": _preds("b", leads=(1, 6))}
    with pytest.raises(ValueError, match="lead"):
        build_station(frames, "古亭", 2025)


def test_build_station_only_carries_whitelisted_fields():
    out = build_station(_frames(), "古亭", 2025)
    assert set(out) == {"station", "station_en", "region", "year", "start", "leads", "metrics"}
    assert set(out["leads"]["1"]) == {"y", "models"}
    assert set(out["leads"]["1"]["models"]["a"]) == {"q10", "q50", "q90"}


def test_station_file_uses_ascii_names():
    assert station_file(2025, "臺南") == "2025/Tainan.json"


def test_build_index_lists_stations_models_and_years():
    index = build_index({"a": "模型 A"}, (2025,))
    assert index["models"] == [{"id": "a", "label": "模型 A"}]
    assert index["years"] == [2025]
    assert index["leads"] == [1, 6, 24]
    gu = next(s for s in index["stations"] if s["name"] == "古亭")
    assert gu["region_zh"] == "北部" and gu["file"] == {"2025": "2025/Guting.json"}
    assert len(index["stations"]) == 12
    assert gu["lat"] == pytest.approx(25.020608)


def test_load_station_frames_reads_each_model(tmp_path):
    for m in MODELS:
        write_predictions(_preds(m), tmp_path / f"古亭__{m}.csv.gz")
    frames = load_station_frames("古亭", MODELS, tmp_path)
    assert list(frames) == list(MODELS)
    assert len(frames["a"]) == 18


def test_load_station_frames_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="古亭__a"):
        load_station_frames("古亭", ("a",), tmp_path)


def test_export_all_writes_strict_json(tmp_path):
    pred_dir, out_dir = tmp_path / "pred", tmp_path / "out"
    pred_dir.mkdir()
    for m in MODELS:
        write_predictions(_preds(m), pred_dir / f"古亭__{m}.csv.gz")
    written = export_all({2025: pred_dir}, out_dir, {"a": "A", "b": "B"}, stations=("古亭",))
    assert (out_dir / "index.json") in written
    station = json.loads((out_dir / "2025/Guting.json").read_text(encoding="utf-8"))
    assert station["station_en"] == "Guting"
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in index["stations"]] == ["古亭"]
    # strict JSON: NaN/Infinity would break JSON.parse in the browser
    for path in written:
        assert "NaN" not in path.read_text(encoding="utf-8")
