"""Turn cached raw CSVs into one processed hourly frame per station.

Run `uv run python -m pm25.data.build` after downloading.
"""

from pathlib import Path

import pandas as pd

from pm25.config import (
    COVARIATES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STATIONS,
    TARGET,
    YEARS,
)
from pm25.data.preprocess import preprocess_station, records_to_frame


def build_station(raw_dir: Path, station: str, years: tuple[int, ...] = YEARS) -> pd.DataFrame:
    paths = [Path(raw_dir) / f"{station}_{year}.csv" for year in years]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"raw files not found: {missing}")
    raw = pd.concat([pd.read_csv(p, dtype=str) for p in paths], ignore_index=True)
    return preprocess_station(records_to_frame(raw.to_dict("records")))


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year hour count and fraction of hours still missing after interpolation."""
    columns = [TARGET, *COVARIATES]
    grouped = df[columns].isna().groupby(df.index.year)
    return grouped.mean().add_suffix("_missing").assign(hours=grouped.size())


def main() -> None:  # pragma: no cover - file IO entry point
    out_dir = Path(PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for station in (s for group in STATIONS.values() for s in group):
        df = build_station(Path(RAW_DATA_DIR), station)
        df.to_pickle(out_dir / f"{station}.pkl")
        reports.append(missing_report(df).assign(station=station))
    summary = pd.concat(reports).reset_index(names="year")
    summary.to_csv(out_dir / "missing_report.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
