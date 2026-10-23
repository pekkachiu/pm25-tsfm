from unittest.mock import patch

from pm25.models.device import default_device


def test_env_override(monkeypatch):
    monkeypatch.setenv("PM25_DEVICE", "cpu")
    assert default_device() == "cpu"


def test_uses_cuda_when_available(monkeypatch):
    monkeypatch.delenv("PM25_DEVICE", raising=False)
    with patch("torch.cuda.is_available", return_value=True):
        assert default_device() == "cuda"


def test_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("PM25_DEVICE", raising=False)
    with patch("torch.cuda.is_available", return_value=False):
        assert default_device() == "cpu"
