"""Export forecasts for the static web demo (GitHub Pages, served from docs/).

Only whitelisted fields of the derived forecasts are written: station names and
coordinates (public MOENV metadata), observed PM2.5 and model quantiles. Each station
becomes one JSON file with dense hourly arrays (null where a lead has no target), so the
browser loads a single station at a time.

    uv run python -m pm25.web.export
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    HORIZONS,
    PREDICTIONS_DIR,
    REGION_ZH,
    STATION_COORDS,
    STATION_EN,
    STATIONS,
    TEST_YEARS,
    WEB_DATA_DIR,
    WEB_DECIMALS,
    WEB_MODELS,
)
from pm25.data.io import read_predictions
from pm25.paths import for_year

VALUE_COLUMNS = ("y", "q10", "q50", "q90")
QUANTILES = ("q10", "q50", "q90")
KEY = ["lead", "target_time"]
METRIC_DECIMALS = 4
REGION_OF = {s: region for region, stations in STATIONS.items() for s in stations}
ALL_STATIONS = tuple(s for stations in STATIONS.values() for s in stations)


def station_file(year: int, station: str) -> str:
    """Relative data path; ASCII names avoid URL-encoding issues on static hosts."""
    return f"{year}/{STATION_EN[station]}.json"


def load_station_frames(station: str, models, pred_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for model in models:
        path = Path(pred_dir) / f"{station}__{model}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"missing predictions: {path}")
        frames[model] = read_predictions(path)
    return frames


def _validate(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Index each model by (lead, target_time) and check they share targets and truth."""
    indexed = {}
    for model, df in frames.items():
        if df[list(VALUE_COLUMNS)].isna().any().any():
            raise ValueError(f"model {model}: missing values in {VALUE_COLUMNS}")
        if set(df["lead"]) != set(HORIZONS):
            raise ValueError(f"model {model}: lead set {sorted(set(df['lead']))} != {HORIZONS}")
        indexed[model] = df.set_index(KEY).sort_index()
    reference_model, reference = next(iter(indexed.items()))
    for model, df in indexed.items():
        if not df.index.equals(reference.index):
            raise ValueError(f"model {model}: targets differ from {reference_model}")
        if not np.allclose(df["y"].to_numpy(), reference["y"].to_numpy()):
            raise ValueError(f"model {model}: observed y differs from {reference_model}")
    return indexed


def _compact(values: np.ndarray) -> list:
    """Rounded floats, whole numbers as int, NaN as null (strict JSON)."""
    out = []
    for value in np.round(values, WEB_DECIMALS):
        if np.isnan(value):
            out.append(None)
        elif float(value).is_integer():
            out.append(int(value))
        else:
            out.append(float(value))
    return out


def _dense(series: pd.Series, positions: np.ndarray, size: int) -> list:
    dense = np.full(size, np.nan)
    dense[positions] = series.to_numpy(dtype=float)
    return _compact(dense)


def _metrics(df: pd.DataFrame) -> dict:
    """MAE and 80% coverage per lead, from unrounded forecasts."""
    out = {}
    for lead, group in df.groupby(level="lead"):
        covered = (group["q10"] <= group["y"]) & (group["y"] <= group["q90"])
        out[str(lead)] = {
            "mae": round(float((group["y"] - group["q50"]).abs().mean()), METRIC_DECIMALS),
            "coverage80": round(float(covered.mean()), METRIC_DECIMALS),
            "n": int(len(group)),
        }
    return out


def build_station(frames: dict[str, pd.DataFrame], station: str, year: int) -> dict:
    indexed = _validate(frames)
    reference = next(iter(indexed.values()))
    times = reference.index.get_level_values("target_time")
    start = times.min()
    size = int((times.max() - start) / pd.Timedelta(hours=1)) + 1
    leads = {}
    for lead in HORIZONS:
        ref_lead = reference.xs(lead, level="lead")
        positions = ((ref_lead.index - start) / pd.Timedelta(hours=1)).to_numpy().astype(int)
        leads[str(lead)] = {
            "y": _dense(ref_lead["y"], positions, size),
            "models": {
                model: {q: _dense(df.xs(lead, level="lead")[q], positions, size)
                        for q in QUANTILES}
                for model, df in indexed.items()
            },
        }
    return {
        "station": station,
        "station_en": STATION_EN[station],
        "region": REGION_OF[station],
        "year": int(year),
        "start": start.isoformat(),
        "leads": leads,
        "metrics": {model: _metrics(df) for model, df in indexed.items()},
    }


def build_index(models: dict[str, str], years, stations=ALL_STATIONS,
                metrics: dict | None = None) -> dict:
    metrics = metrics or {}
    entries = []
    for station in stations:
        lat, lon = STATION_COORDS[station]
        entry = {
            "name": station,
            "en": STATION_EN[station],
            "region": REGION_OF[station],
            "region_zh": REGION_ZH[REGION_OF[station]],
            "lat": lat,
            "lon": lon,
            "file": {str(y): station_file(y, station) for y in years},
        }
        if station in metrics:
            entry["metrics"] = metrics[station]
        entries.append(entry)
    return {
        "models": [{"id": m, "label": label} for m, label in models.items()],
        "years": [int(y) for y in years],
        "leads": list(HORIZONS),
        "stations": entries,
    }


def _write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path


def export_all(pred_dirs: dict[int, Path], out_dir: Path, models: dict[str, str],
               stations=ALL_STATIONS) -> list[Path]:
    out_dir = Path(out_dir)
    written, metrics = [], {}
    for year, pred_dir in pred_dirs.items():
        for station in stations:
            frames = load_station_frames(station, tuple(models), Path(pred_dir))
            payload = build_station(frames, station, year)
            metrics.setdefault(station, {})[str(year)] = payload["metrics"]
            written.append(_write_json(payload, out_dir / station_file(year, station)))
    index = build_index(models, tuple(pred_dirs), stations, metrics)
    written.append(_write_json(index, out_dir / "index.json"))
    return written


def main() -> None:  # pragma: no cover - file IO entry point
    pred_dirs = {year: Path(for_year(PREDICTIONS_DIR, year)) for year in TEST_YEARS}
    written = export_all(pred_dirs, Path(WEB_DATA_DIR), WEB_MODELS)
    total = sum(p.stat().st_size for p in written)
    print(f"wrote {len(written)} files, {total / 1e6:.1f} MB -> {WEB_DATA_DIR}")


if __name__ == "__main__":  # pragma: no cover
    main()
