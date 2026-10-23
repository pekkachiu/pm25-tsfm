"""Two-layer LSTM baseline trained with the pinball (quantile) loss.

Uses the same 48 h inputs as XGBoost (target, covariates, calendar), predicts the
selected leads directly, and fills NaN inputs with the training mean (0 after scaling).
Known-future covariates (e.g. NWP wind over the forecast window) are flattened and joined
to the last hidden state before the output layer.
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from pm25.config import COVARIATES, HORIZONS, QUANTILE_LEVELS, TARGET
from pm25.evaluation.rolling import ForecastInput
from pm25.models.device import default_device
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
    def __init__(self, n_inputs: int, hidden_size: int, n_leads: int, n_quantiles: int,
                 n_future: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden_size, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_size + n_future, n_leads * n_quantiles)
        self.out_shape = (n_leads, n_quantiles)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, channels); future: (batch, n_future), possibly zero-width."""
        hidden, _ = self.lstm(x)
        return self.head(torch.cat([hidden[:, -1], future], dim=1)).view(-1, *self.out_shape)


class LSTMModel:
    def __init__(self, leads: tuple[int, ...] = HORIZONS,
                 covariates: tuple[str, ...] = COVARIATES,
                 future_covariates: tuple[str, ...] = (), name: str = "lstm",
                 quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
                 hidden_size: int = 64, batch_size: int = 256, learning_rate: float = 1e-3,
                 max_epochs: int = 50, patience: int = 5, seed: int = 0):
        self.name = name
        self._leads = leads
        self._columns = (TARGET, *covariates)
        self._future = future_covariates
        self._quantiles = quantile_levels
        self._hidden_size = hidden_size
        self._batch_size = batch_size
        self._lr = learning_rate
        self._max_epochs = max_epochs
        self._patience = patience
        self._seed = seed
        self._net: _Net | None = None
        self._mean = self._std = None
        self._future_mean = self._future_std = None
        self._device = None

    def _scaled_steps(self, frame: pd.DataFrame) -> np.ndarray:
        """(time, channels) with numeric channels standardised and NaN set to 0."""
        values = (frame[list(self._columns)].to_numpy(dtype=float) - self._mean) / self._std
        return np.nan_to_num(np.hstack([values, _calendar(frame.index)])).astype(np.float32)

    def _scaled_future(self, frame: pd.DataFrame) -> np.ndarray:
        """(time, future covariates) standardised with NaN set to 0."""
        values = frame[list(self._future)].to_numpy(dtype=float).reshape(len(frame), -1)
        return np.nan_to_num((values - self._future_mean) / self._future_std).astype(np.float32)

    def _training_tensors(self, train: pd.DataFrame):
        steps = self._scaled_steps(train)
        horizon = max(self._leads)
        n_origins = len(steps) - LAGS + 1 - horizon
        x = sliding_window_view(steps, LAGS, axis=0)[:n_origins]
        # future window of origin LAGS-1+k covers rows LAGS+k .. LAGS+k+horizon-1
        future = sliding_window_view(self._scaled_future(train), horizon, axis=0)[
            LAGS:LAGS + n_origins].reshape(n_origins, -1)
        target = train[TARGET].to_numpy(dtype=float)
        y = np.stack([target[LAGS - 1 + lead:len(target) - horizon + lead]
                      for lead in self._leads], axis=1)
        keep = ~np.isnan(y).any(axis=1)
        y_scaled = (y[keep] - self._mean[0]) / self._std[0]
        return (torch.from_numpy(np.ascontiguousarray(x[keep].transpose(0, 2, 1))),
                torch.from_numpy(np.ascontiguousarray(future[keep])),
                torch.from_numpy(y_scaled.astype(np.float32)))

    def _run_epoch(self, net, optimiser, x, f, y, q, train: bool) -> float:
        net.train(train)
        order = torch.randperm(len(x)) if train else torch.arange(len(x))
        total = 0.0
        for start in range(0, len(x), self._batch_size):
            idx = order[start:start + self._batch_size]
            xb, fb, yb = (t[idx].to(self._device) for t in (x, f, y))
            with torch.set_grad_enabled(train):
                loss = pinball_loss(net(xb, fb), yb, q)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * len(idx)
        return total / len(x)

    def fit(self, train: pd.DataFrame) -> "LSTMModel":
        """Train with early stopping on the chronologically last 10% of `train`."""
        torch.manual_seed(self._seed)
        self._device = torch.device(default_device())
        numeric = train[list(self._columns)].to_numpy(dtype=float)
        self._mean = np.nanmean(numeric, axis=0)
        self._std = np.nanstd(numeric, axis=0) + 1e-6
        future = train[list(self._future)].to_numpy(dtype=float).reshape(len(train), -1)
        self._future_mean = np.nanmean(future, axis=0) if self._future else np.zeros(0)
        self._future_std = (np.nanstd(future, axis=0) + 1e-6) if self._future else np.ones(0)
        x, f, y = self._training_tensors(train)
        split = int(len(x) * (1 - VALIDATION_FRACTION))
        q = torch.tensor(self._quantiles, dtype=torch.float32, device=self._device)
        net = _Net(x.shape[-1], self._hidden_size, len(self._leads),
                   len(self._quantiles), f.shape[-1]).to(self._device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self._lr)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self._max_epochs):
            self._run_epoch(net, optimiser, x[:split], f[:split], y[:split], q, train=True)
            val = self._run_epoch(net, optimiser, x[split:], f[split:], y[split:], q,
                                  train=False)
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
        span = max(self._leads)
        f = torch.from_numpy(np.stack([self._scaled_future(i.future.iloc[:span]).reshape(-1)
                                       for i in inputs]))
        b = self._batch_size
        with torch.no_grad():
            pred = torch.cat([self._net(x[i:i + b].to(self._device),
                                        f[i:i + b].to(self._device)).cpu()
                              for i in range(0, len(x), b)]).numpy()
        pred = pred * self._std[0] + self._mean[0]  # (n, lead, q)
        out = np.full((len(inputs), len(quantile_levels), horizon), np.nan)
        for j, lead in enumerate(self._leads):
            out[:, :, lead - 1] = np.sort(pred[:, j, :], axis=1)
        return out
