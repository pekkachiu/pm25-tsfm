from unittest.mock import MagicMock

import pytest

import pandas as pd

from pm25.data.download import MoenvClient, download_station_year, get_api_key, year_filter


def _response(records):
    r = MagicMock()
    r.json.return_value = records
    r.raise_for_status.return_value = None
    return r


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("MOENV_API_KEY", "abc")
    assert get_api_key() == "abc"


def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("MOENV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MOENV_API_KEY"):
        get_api_key()


def test_fetch_paginates_until_short_page():
    session = MagicMock()
    session.get.side_effect = [
        _response([{"a": 1}, {"a": 2}]),
        _response([{"a": 3}]),
    ]
    client = MoenvClient(api_key="k", session=session, page_size=2, sleep_seconds=0)
    records = client.fetch("aqx_p_488", filters="SiteName,EQ,古亭")
    assert [r["a"] for r in records] == [1, 2, 3]
    first_params = session.get.call_args_list[0].kwargs["params"]
    assert first_params["api_key"] == "k"
    assert first_params["offset"] == 0 and first_params["limit"] == 2
    assert first_params["filters"] == "SiteName,EQ,古亭"
    assert session.get.call_args_list[1].kwargs["params"]["offset"] == 2


def test_fetch_non_json_error_message_raises():
    session = MagicMock()
    bad = MagicMock()
    bad.raise_for_status.return_value = None
    bad.json.side_effect = ValueError("no json")
    bad.text = "api_key 不存在。"
    session.get.return_value = bad
    client = MoenvClient(api_key="k", session=session, sleep_seconds=0)
    with pytest.raises(RuntimeError, match="api_key"):
        client.fetch("aqx_p_488")


def test_fetch_accepts_wrapped_records_payload():
    session = MagicMock()
    wrapped = MagicMock()
    wrapped.raise_for_status.return_value = None
    wrapped.json.return_value = {"records": [{"a": 1}]}
    session.get.return_value = wrapped
    client = MoenvClient(api_key="k", session=session, sleep_seconds=0)
    assert client.fetch("aqx_p_488") == [{"a": 1}]


def test_year_filter_uses_minute_precision_string_bounds():
    # The API compares dates as strings, so bounds must sort around "YYYY-MM-DD HH:MM".
    f = year_filter("古亭", 2021)
    assert f == ("sitename,EQ,古亭|datacreationdate,GR,2020-12-31 23:59"
                 "|datacreationdate,LT,2021-12-31 23:59")


def test_download_station_year_writes_csv(tmp_path):
    client = MagicMock()
    client.fetch.return_value = [{"sitename": "古亭", "pm2.5": "10",
                                  "datacreationdate": "2021-01-01 01:00"}]
    path = download_station_year(client, "古亭", 2021, tmp_path)
    assert path == tmp_path / "古亭_2021.csv"
    df = pd.read_csv(path, dtype=str)
    assert df.loc[0, "pm2.5"] == "10"
    assert client.fetch.call_args.kwargs["sort"] == "datacreationdate asc"


def test_download_station_year_skips_existing(tmp_path):
    (tmp_path / "古亭_2021.csv").write_text("x\n1\n")
    client = MagicMock()
    download_station_year(client, "古亭", 2021, tmp_path)
    client.fetch.assert_not_called()


def test_download_station_year_empty_raises(tmp_path):
    client = MagicMock()
    client.fetch.return_value = []
    with pytest.raises(RuntimeError, match="no records"):
        download_station_year(client, "古亭", 2021, tmp_path)
    assert not (tmp_path / "古亭_2021.csv").exists()
