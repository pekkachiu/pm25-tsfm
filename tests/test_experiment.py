import numpy as np
import pandas as pd
import pytest

from pm25.data.io import read_predictions
from pm25.experiment import make_models, prediction_frame, run_station
from pm25.models.persistence import PersistenceModel


@pytest.fixture
def station_frame(rng):
    idx = pd.date_range("2024-12-01", "2025-01-10 23:00", freq="h", tz="Asia/Taipei")
    return pd.DataFrame({"pm25": rng.uniform(5, 40, len(idx)),
                         "wind_speed": rng.uniform(0, 3, len(idx))}, index=idx)


def test_prediction_frame_long_format():
    origins = list(pd.date_range("2025-01-01", periods=2, freq="h", tz="Asia/Taipei"))
    targets = np.arange(2 * 24, dtype=float).reshape(2, 24)
    q = np.stack([targets - 1, targets, targets + 1], axis=1)
    out = prediction_frame("古亭", "m", origins, targets, q, leads=(1, 24))
    assert len(out) == 4
    row = out[(out["origin"] == origins[1]) & (out["lead"] == 24)].iloc[0]
    assert row["y"] == 47.0 and row["q10"] == 46.0 and row["q50"] == 47.0 and row["q90"] == 48.0
    assert row["target_time"] == origins[1] + pd.Timedelta(hours=24)
    assert set(out["station"]) == {"古亭"} and set(out["model"]) == {"m"}


def test_run_station_writes_and_skips_existing(station_frame, tmp_path):
    paths = run_station("古亭", station_frame, [PersistenceModel()], tmp_path,
                        test_year=2025, context_length=48)
    assert [p.name for p in paths] == ["古亭__persistence.csv.gz"]
    preds = read_predictions(paths[0])
    assert preds["origin"].min().year == 2025
    assert np.allclose(preds["q50"], preds["q10"])  # persistence is a point forecast
    mtime = paths[0].stat().st_mtime
    run_station("古亭", station_frame, [PersistenceModel()], tmp_path,
                test_year=2025, context_length=48)
    assert paths[0].stat().st_mtime == mtime


def test_run_station_fits_trainable_models_on_years_before_test(station_frame, tmp_path):
    class Spy(PersistenceModel):
        name = "spy"
        seen = None

        def fit(self, train):
            Spy.seen = train
            return self

    run_station("古亭", station_frame, [Spy()], tmp_path, test_year=2025, context_length=48)
    assert Spy.seen.index.max().year == 2024


def test_make_models_rejects_unknown_family():
    with pytest.raises(ValueError, match="family"):
        make_models("nope")


def test_make_models_baseline():
    assert [m.name for m in make_models("baseline")] == ["persistence"]


def test_make_models_nwp_families():
    assert [m.name for m in make_models("nwp_bestmatch")] == ["chronos2_wind_bestmatch"]
    assert [m.name for m in make_models("nwp_torch")] == ["lstm_nwp", "chronos2_wind_nwp"]


def test_family_nwp_sources():
    from pm25.experiment import FAMILY_NWP

    assert FAMILY_NWP == {"nwp_bestmatch": "best_match", "nwp_xgboost": "jma_gsm",
                          "nwp_torch": "jma_gsm", "finetuned": "jma_gsm",
                          "spatial_xgboost": "jma_gsm", "spatial_chronos": "jma_gsm"}


def test_make_models_spatial_chronos_inputs():
    models = {m.name: m for m in make_models("spatial_chronos")}
    assert list(models) == ["chronos2_spatial", "chronos2_spatial_region",
                            "chronos2_spatial_all", "chronos2_spatial_nwp"]
    assert models["chronos2_spatial"]._past[-3:] == ("pm25_nearest1", "pm25_nearest2",
                                                     "pm25_nearest3")
    assert models["chronos2_spatial_region"]._past[-1] == "pm25_region2"
    assert models["chronos2_spatial_all"]._past[-1] == "pm25_all11"
    assert models["chronos2_spatial_nwp"]._future == ("fc_wind_speed", "fc_wind_sin",
                                                      "fc_wind_cos")
    assert models["chronos2_spatial"]._future == ()


def _write_processed(directory, stations, idx):
    from pm25.data.io import write_station

    directory.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(stations):
        write_station(pd.DataFrame({"pm25": float(i), "station": s}, index=idx), directory, s)


def test_load_station_attaches_nwp_and_neighbours(tmp_path):
    from pm25.config import STATIONS
    from pm25.data.io import write_station
    from pm25.experiment import load_station

    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="Asia/Taipei")
    stations = [s for group in STATIONS.values() for s in group]
    _write_processed(tmp_path / "processed", stations, idx)
    (tmp_path / "nwp" / "jma_gsm").mkdir(parents=True)
    write_station(pd.DataFrame({"fc_wind_speed": 1.5}, index=idx[:2]),
                  tmp_path / "nwp" / "jma_gsm", "古亭")
    df = load_station("古亭", "spatial_chronos", tmp_path / "processed", tmp_path / "nwp")
    assert "station" not in df.columns
    assert df["fc_wind_speed"].isna().tolist() == [False, False, True]
    assert (df["pm25_nearest1"] == stations.index("板橋")).all()
    assert df.filter(like="pm25_all").shape[1] == 11
    plain = load_station("古亭", "baseline", tmp_path / "processed")
    assert plain.columns.tolist() == ["pm25"]


def test_make_models_finetuned_uses_existing_checkpoints(tmp_path, monkeypatch):
    import pm25.experiment as experiment

    monkeypatch.setattr(experiment, "FINETUNED_DIR", str(tmp_path))
    assert make_models("finetuned") == []
    (tmp_path / "chronos2_ft_nwp").mkdir()
    models = make_models("finetuned")
    assert [m.name for m in models] == ["chronos2_ft_nwp"]
    assert models[0]._future == ("fc_wind_speed", "fc_wind_sin", "fc_wind_cos")


def test_make_models_finetuned_reads_year_specific_checkpoints(tmp_path, monkeypatch):
    import pm25.experiment as experiment

    monkeypatch.setattr(experiment, "FINETUNED_DIR", str(tmp_path / "finetuned"))
    (tmp_path / "finetuned" / "chronos2_ft_uni").mkdir(parents=True)
    assert make_models("finetuned", test_year=2024) == []
    (tmp_path / "finetuned_2024" / "chronos2_ft_cov").mkdir(parents=True)
    assert [m.name for m in make_models("finetuned", test_year=2024)] == ["chronos2_ft_cov"]


def test_check_family_year_rejects_bestmatch_before_its_archive():
    from pm25.experiment import check_family_year

    check_family_year("nwp_bestmatch", 2025)
    check_family_year("chronos", 2024)
    with pytest.raises(ValueError, match="best_match"):
        check_family_year("nwp_bestmatch", 2024)
