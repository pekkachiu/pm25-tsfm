"""Pick the torch device: $PM25_DEVICE if set, else CUDA when available, else CPU.

MPS is not used by default: Chronos and the small LSTM gain little from it on Apple Silicon.
"""

import os

DEVICE_ENV = "PM25_DEVICE"


def default_device() -> str:
    override = os.environ.get(DEVICE_ENV)
    if override:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
