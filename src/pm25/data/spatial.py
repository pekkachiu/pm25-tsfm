"""Neighbouring stations' PM2.5 as covariates (phase 4).

Neighbour readings at the origin hour are published in real time, so they are past
covariates only; their future is never used.
"""

import math

import pandas as pd

from pm25.config import SPATIAL_K, SPATIAL_SCOPES, STATION_COORDS, STATIONS, TARGET

EARTH_RADIUS_KM = 6371.0
_REGION = {s: region for region, group in STATIONS.items() for s in group}


def distance_km(a: str, b: str) -> float:
    """Great-circle (haversine) distance between two stations."""
    (lat1, lon1), (lat2, lon2) = STATION_COORDS[a], STATION_COORDS[b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def neighbours(station: str, scope: str, k: int = SPATIAL_K) -> tuple[str, ...]:
    """Other stations in `scope`, nearest first."""
    if station not in _REGION:
        raise ValueError(f"unknown station {station!r}")
    if scope not in SPATIAL_SCOPES:
        raise ValueError(f"unknown scope {scope!r}; choose from {SPATIAL_SCOPES}")
    others = sorted((s for s in _REGION if s != station), key=lambda s: distance_km(station, s))
    if scope == "nearest":
        return tuple(others[:k])
    if scope == "region":
        return tuple(s for s in others if _REGION[s] == _REGION[station])
    return tuple(others)


def spatial_columns(scope: str, k: int = SPATIAL_K) -> tuple[str, ...]:
    """Column names of a scope's neighbours (same for every station: rank, not name)."""
    station = next(iter(_REGION))
    return tuple(f"{TARGET}_{scope}{i + 1}" for i in range(len(neighbours(station, scope, k))))


def attach_neighbours(df: pd.DataFrame, station: str, frames: dict[str, pd.DataFrame],
                      scopes: tuple[str, ...] = SPATIAL_SCOPES) -> pd.DataFrame:
    """New frame with each scope's neighbour PM2.5 aligned to df's hourly index."""
    columns = {}
    for scope in scopes:
        for column, other in zip(spatial_columns(scope), neighbours(station, scope)):
            if other not in frames:
                raise ValueError(f"missing frame for neighbour {other}")
            columns[column] = frames[other][TARGET].reindex(df.index)
    return df.assign(**columns)
