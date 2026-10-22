"""Paginated client for the MOENV open-data API (data.moenv.gov.tw).

Run `uv run --env-file .env python -m pm25.data.download` to cache raw data per station-year.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

from pm25.config import (
    MOENV_API_BASE,
    MOENV_API_KEY_ENV,
    MOENV_DATASET,
    RAW_DATA_DIR,
    STATIONS,
    YEARS,
)


def get_api_key() -> str:
    key = os.environ.get(MOENV_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set the {MOENV_API_KEY_ENV} environment variable (see .env)")
    return key


class MoenvClient:
    def __init__(self, api_key: str, session: requests.Session | None = None,
                 page_size: int = 1000, sleep_seconds: float = 0.5, timeout: float = 120):
        # The API caps pages at 1000 records regardless of the requested limit.
        self._api_key = api_key
        self._session = session or requests.Session()
        self._page_size = page_size
        self._sleep_seconds = sleep_seconds
        self._timeout = timeout

    def _get_page(self, dataset: str, offset: int, filters: str | None,
                  sort: str | None) -> list[dict]:
        params = {"api_key": self._api_key, "format": "json",
                  "offset": offset, "limit": self._page_size}
        if filters:
            params["filters"] = filters
        if sort:
            params["sort"] = sort
        response = self._session.get(f"{MOENV_API_BASE}/{dataset}", params=params,
                                     timeout=self._timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            # The API answers errors (e.g. bad key) with a plain-text body and HTTP 200.
            raise RuntimeError(f"MOENV API error: {response.text[:200]}") from exc
        return payload if isinstance(payload, list) else payload.get("records", [])

    def fetch(self, dataset: str, filters: str | None = None,
              sort: str | None = None) -> list[dict]:
        """Fetch every record of a dataset, following offset pagination."""
        records: list[dict] = []
        offset = 0
        while True:
            page = self._get_page(dataset, offset, filters, sort)
            records = [*records, *page]
            if len(page) < self._page_size:
                return records
            offset += self._page_size
            time.sleep(self._sleep_seconds)


def year_filter(station: str, year: int) -> str:
    """Filter for one station-year. Dates are compared as strings by the API,
    so minute-precision bounds are used to include hour 00:00 correctly."""
    return (f"sitename,EQ,{station}"
            f"|datacreationdate,GR,{year - 1}-12-31 23:59"
            f"|datacreationdate,LT,{year}-12-31 23:59")


def download_station_year(client: MoenvClient, station: str, year: int,
                          out_dir: Path) -> Path:
    """Download one station-year to `<out_dir>/<station>_<year>.csv` (skipped if cached)."""
    path = Path(out_dir) / f"{station}_{year}.csv"
    if path.exists():
        return path
    records = client.fetch(MOENV_DATASET, filters=year_filter(station, year),
                           sort="datacreationdate asc")
    if not records:
        raise RuntimeError(f"no records for {station} {year}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def main() -> None:  # pragma: no cover - network entry point
    client = MoenvClient(api_key=get_api_key())
    stations = [s for group in STATIONS.values() for s in group]
    for station in stations:
        for year in YEARS:
            path = download_station_year(client, station, year, Path(RAW_DATA_DIR))
            print(f"{station} {year} -> {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
