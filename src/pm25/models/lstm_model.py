"""Two-layer LSTM baseline trained with the pinball (quantile) loss.

Uses the same 48 h inputs as XGBoost (target, covariates, calendar), predicts the
selected leads directly, and fills NaN inputs with the training mean (0 after scaling).
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from pm25.config import COVARIATES, HORIZONS, QUANTILE_LEVELS, TARGET
from pm25.evaluation.rolling import ForecastInput
from pm25.models.features import LAGS

VALIDATION_FRACTION = 0.1


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred: (batch, lead, quantile); y: (batch, lead)."""
    diff = y.unsqueeze(-1) - pred
    return torch.mean(torch.maximum(quantiles * diff, (quantiles - 1) * diff))


def _calendar(index: pd.DatetimeIndex) -> np.ndarray:
    angles = [2 * np.pi * index.hour / 24, 2 * np.pi * index.day_of_week / 7,
              2 * np.pi * (index.month - 1) / 12]
    return np.stack([f(a) for a in angles for f in (np.sin, np.cos)], axis=1)


class _Net(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, n_leads: int, n_quantiles: int):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_size, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_size, n_leads * n_quantiles)
        self.out_shape = (n_leads, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        return self.head(hidden[:, -1]).view(-1, *self.out_shape)


class LSTMModel:
    name = "lstm"

    def __init__(self, leads: tuple[int, ...] = HORIZONS,
                 covariates: tuple[str, ...] = COVARIATES,
                 quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
                 hidden_size: int = 64, batch_size: int = 256, learning_rate: float = 1e-3,
                 max_epochs: int = 50, patience: int = 5, seed: int = 0):
        self._leads = leads
        self._columns = (TARGET, *covariates)
        self._quantiles = quantile_levels
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._lr = learning_rate
        self._max_epochs = max_epochs
        self._patience = patience
        self._seed = seed
        self._net: _Net | None = None
        self._mean = self._std = None

    def _scaled_steps(self, frame: pd.DataFrame) -> np.ndarray:
        """(time, channels) with numeric channels standardised and NaN set to 0."""
        values = (frame[list(self._columns)].to_numpy(dtype=float) - self._mean) / self._std
        return np.nan_to_num(np.hstack([values, _calendar(frame.index)])).astype(np.float32)

    def _training_tensors(self, train: pd.DataFrame):
        steps = self._scaled_steps(train)
        horizon = max(self._leads)
        x = sliding_window_view(steps, LAGS, axis=0)[:len(steps) - LAGS + 1 - horizon]
        target = train[TARGET].to_numpy(dtype=float)
        y = np.stack([target[LAGS - 1 + lead:len(target) - horizon + lead]
                      for lead in self._leads], axis=1)
        keep = ~np.isnan(y).any(axis=1)
        y_scaled = (y[keep] - self._mean[0]) / self._std[0]
        return (torch.from_numpy(np.ascontiguousarray(x[keep].transpose(0, 2, 1))),
                torch.from_numpy(y_scaled.astype(np.float32)))

    def _run_epoch(self, net, optimiser, x, y, q, train: bool) -> float:
        net.train(train)
        order = torch.randperm(len(x)) if train else torch.arange(len(x))
        total = 0.0
        for start in range(0, len(x), self._batch_size):
            idx = order[start:start + self._batch_size]
            with torch.set_grad_enabled(train):
                loss = pinball_loss(net(x[idx]), y[idx], q)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * len(idx)
        return total / len(x)

    def fit(self, train: pd.DataFrame) -> "LSTMModel":
        """Train with early stopping on the chronologically last 10% of `train`."""
        torch.manual_seed(self._seed)
        numeric = train[list(self._columns)].to_numpy(dtype=float)
        self._mean = np.nanmean(numeric, axis=0)
        self._std = np.nanstd(numeric, axis=0) + 1e-6
        x, y = self._training_tensors(train)
        split = int(len(x) * (1 - VALIDATION_FRACTION))
        q = torch.tensor(self._quantiles, dtype=torch.float32)
        net = _Net(x.shape[-1], self._hidden_size, len(self._leads), len(self._quantiles))
        optimiser = torch.optim.Adam(net.parameters(), lr=self._lr)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self._max_epochs):
            self._run_epoch(net, optimiser, x[:split], y[:split], q, train=True)
            val = self._run_epoch(net, optimiser, x[split:], y[split:], q, train=False)
            if val < best:
                best, stale = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self._patience:
                    break
        net.load_state_dict(best_state)
        self._net = net.eval()
        return self

    def predict(self, inputs: list[ForecastInput], horizon: int,
                quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> np.ndarray:
        """(n, quantile, horizon); leads without a trained output are NaN."""
        if self._net is None:
            raise RuntimeError("call fit() before predict()")
        if tuple(quantile_levels) != tuple(self._quantiles):
            raise ValueError("quantile_levels must match those used in fit()")
        if max(self._leads) > horizon:
            raise ValueError(f"trained lead {max(self._leads)} exceeds horizon {horizon}")
        x = torch.from_numpy(np.stack([self._scaled_steps(i.context.iloc[-LAGS:]) for i in inputs]))
        with torch.no_grad():
            pred = self._net(x).numpy() * self._std[0] + self._mean[0]  # (n, lead, q)
        out = np.full((len(inputs), len(quantile_levels), horizon), np.nan)
        for j, lead in enumerate(self._leads):
            out[:, :, lead - 1] = np.sort(pred[:, j, :], axis=1)
        return out
