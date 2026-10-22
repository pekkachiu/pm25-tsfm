"""Paginated client for the MOENV open-data API (data.moenv.gov.tw)."""

import os
import time

import requests

from pm25.config import MOENV_API_BASE, MOENV_API_KEY_ENV


def get_api_key() -> str:
    key = os.environ.get(MOENV_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set the {MOENV_API_KEY_ENV} environment variable (see .env)")
    return key


class MoenvClient:
    def __init__(self, api_key: str, session: requests.Session | None = None,
                 page_size: int = 1000, sleep_seconds: float = 0.5, timeout: float = 60):
        self._api_key = api_key
        self._session = session or requests.Session()
        self._page_size = page_size
        self._sleep_seconds = sleep_seconds
        self._timeout = timeout

    def _get_page(self, dataset: str, offset: int, filters: str | None) -> list[dict]:
        params = {"api_key": self._api_key, "format": "json",
                  "offset": offset, "limit": self._page_size}
        if filters:
            params["filters"] = filters
        response = self._session.get(f"{MOENV_API_BASE}/{dataset}", params=params,
                                     timeout=self._timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            # The API answers errors (e.g. bad key) with a plain-text body and HTTP 200.
            raise RuntimeError(f"MOENV API error: {response.text[:200]}") from exc
        return payload.get("records", [])

    def fetch(self, dataset: str, filters: str | None = None) -> list[dict]:
        """Fetch every record of a dataset, following offset pagination."""
        records: list[dict] = []
        offset = 0
        while True:
            page = self._get_page(dataset, offset, filters)
            records = [*records, *page]
            if len(page) < self._page_size:
                return records
            offset += self._page_size
            time.sleep(self._sleep_seconds)
