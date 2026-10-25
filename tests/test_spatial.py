import numpy as np
import pandas as pd
import pytest

from pm25.data.spatial import (
    attach_neighbours,
    distance_km,
    neighbours,
    spatial_columns,
)


def test_distance_km_known_pair():
    # Guting-Banqiao are ~7 km apart; distance is symmetric and zero to itself
    assert distance_km("古亭", "板橋") == pytest.approx(7.2, abs=0.5)
    assert distance_km("古亭", "板橋") == pytest.approx(distance_km("板橋", "古亭"))
    assert distance_km("古亭", "古亭") == 0


def test_neighbours_nearest_sorted_by_distance():
    assert neighbours("古亭", "nearest") == ("板橋", "桃園", "忠明")
    # Tainan's three nearest stations are all in Kao-Ping, not its own region
    assert neighbours("臺南", "nearest") == ("前金", "屏東", "小港")


def test_neighbours_region_and_all():
    assert neighbours("臺南", "region") == ("嘉義", "斗六")
    every = neighbours("古亭", "all")
    assert len(every) == 11 and "古亭" not in every and every[0] == "板橋"


def test_neighbours_rejects_unknown_scope_and_station():
    with pytest.raises(ValueError, match="scope"):
        neighbours("古亭", "upwind")
    with pytest.raises(ValueError, match="station"):
        neighbours("台北", "nearest")


def test_spatial_columns():
    assert spatial_columns("nearest") == ("pm25_nearest1", "pm25_nearest2", "pm25_nearest3")
    assert len(spatial_columns("region")) == 2 and len(spatial_columns("all")) == 11


def test_attach_neighbours_aligns_on_index_and_keeps_input():
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="Asia/Taipei")
    df = pd.DataFrame({"pm25": [1.0, 2, 3, 4]}, index=idx)
    others = {s: pd.DataFrame({"pm25": [10.0 * (i + 1)] * 3}, index=idx[1:])
              for i, s in enumerate(neighbours("古亭", "all"))}
    out = attach_neighbours(df, "古亭", others, scopes=("nearest", "region"))
    assert list(df.columns) == ["pm25"]  # input not mutated
    assert out.columns.tolist() == ["pm25", "pm25_nearest1", "pm25_nearest2",
                                    "pm25_nearest3", "pm25_region1", "pm25_region2"]
    assert np.isnan(out["pm25_nearest1"].iloc[0])  # missing hour stays NaN
    assert out["pm25_nearest1"].iloc[1] == 10.0  # 板橋 is the first neighbour
    assert out["pm25_region2"].iloc[1] == 20.0  # 桃園 is the second neighbour


def test_attach_neighbours_requires_every_neighbour_frame():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="Asia/Taipei")
    df = pd.DataFrame({"pm25": [1.0, 2]}, index=idx)
    with pytest.raises(ValueError, match="板橋"):
        attach_neighbours(df, "古亭", {}, scopes=("nearest",))
