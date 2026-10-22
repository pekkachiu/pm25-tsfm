from unittest.mock import MagicMock

import pytest

from pm25.data.download import MoenvClient, get_api_key


def _response(records):
    r = MagicMock()
    r.json.return_value = {"records": records}
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
