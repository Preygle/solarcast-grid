"""LSTM sequence forecaster (PyTorch, CPU).

Per target hour t with issue time tau, the input is the 168-hour sequence ending at tau of
[demand, solar, wind] (z-scored with training statistics, NaN -> 0 after scaling, plus a
missing flag) and hour-of-day sin/cos, followed by a static vector:
target-time calendar (hour, day-of-week, day-of-year harmonics) and the lead (t - tau) / 38.
The target is learnt as y / level (same level feature as the tabular models).
Only data at or before tau enters the network, so the leakage rules match the tabular features.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

SEQ_LEN = 168
INPUT_SERIES = ["demand_mw", "solar_mw", "wind_mw"]


class _Net(nn.Module):
    def __init__(self, n_in: int, n_static: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


class LSTMForecaster:
    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.1, lr: float = 1e-3,
                 batch_size: int = 256, max_epochs: int = 40, patience: int = 6, seed: int = 42):
        self.hidden, self.layers, self.dropout, self.lr = hidden, layers, dropout, lr
        self.batch_size, self.max_epochs, self.patience, self.seed = batch_size, max_epochs, patience, seed

    # ---------- data ----------
    def _prepare_base(self, hourly: pd.DataFrame) -> None:
        self.mu_ = hourly[INPUT_SERIES].mean().to_numpy()
        sd = hourly[INPUT_SERIES].std().to_numpy()
        self.sd_ = np.where(sd > 0, sd, 1.0)

    def _arrays(self, hourly: pd.DataFrame, rows: pd.DatetimeIndex, issue: pd.DatetimeIndex,
                level: np.ndarray):
        vals = hourly[INPUT_SERIES].to_numpy(dtype=np.float32)
        z = (vals - self.mu_) / self.sd_
        miss = np.isnan(z).any(axis=1, keepdims=True).astype(np.float32)
        z = np.nan_to_num(z, nan=0.0)
        hr = hourly.index.hour.to_numpy()
        feats = np.concatenate([z, miss, np.sin(2 * np.pi * hr / 24)[:, None], np.cos(2 * np.pi * hr / 24)[:, None]],
                               axis=1).astype(np.float32)
        pos = hourly.index.get_indexer(issue)
        ok = pos >= SEQ_LEN - 1
        offs = np.arange(-SEQ_LEN + 1, 1)
        seq = feats[np.clip(pos[:, None] + offs[None, :], 0, None)]
        lead = (rows - issue) / pd.Timedelta(hours=1)
        doy = rows.dayofyear.to_numpy()
        static = np.column_stack([
            np.sin(2 * np.pi * rows.hour / 24), np.cos(2 * np.pi * rows.hour / 24),
            np.sin(2 * np.pi * rows.dayofweek / 7), np.cos(2 * np.pi * rows.dayofweek / 7),
            (rows.dayofweek >= 5).astype(float),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
            np.asarray(lead, dtype=float) / 38.0,
        ]).astype(np.float32)
        return seq, static, ok

    # ---------- training ----------
    def _train(self, seq, static, y, seq_v=None, static_v=None, y_v=None, epochs=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(8)
        net = _Net(seq.shape[2], static.shape[1], self.hidden, self.layers, self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()
        S, St, Y = map(torch.from_numpy, (seq, static, y.astype(np.float32)))
        best, best_state, best_ep, bad = np.inf, None, 0, 0
        n_ep = epochs or self.max_epochs
        history = []
        for ep in range(n_ep):
            net.train()
            perm = torch.randperm(len(Y))
            for i in range(0, len(Y), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(S[b], St[b]), Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if seq_v is not None:
                v = float(np.mean(np.abs(self._predict_raw(net, seq_v, static_v) - y_v)))
                history.append(v)
                if v < best - 1e-5:
                    best, best_state, best_ep, bad = v, copy.deepcopy(net.state_dict()), ep + 1, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep, history

    @staticmethod
    def _predict_raw(net, seq, static, bs: int = 2048) -> np.ndarray:
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seq), bs):
                out.append(net(torch.from_numpy(seq[i:i + bs]), torch.from_numpy(static[i:i + bs])).numpy())
        return np.concatenate(out)

    def fit_with_validation(self, hourly, rows, issue, level, y, rows_v, issue_v, level_v, y_v):
        """Train on (rows), early-stop on (rows_v). Returns validation MAE in target units."""
        self._prepare_base(hourly.loc[: rows.max()])
        s, st, ok = self._arrays(hourly, rows, issue, level)
        sv, stv, okv = self._arrays(hourly, rows_v, issue_v, level_v)
        yr, yrv = y / level, y_v / level_v
        m, mv = ok & np.isfinite(yr), okv & np.isfinite(yrv)
        self.net_, self.best_epochs_, self.history_ = self._train(s[m], st[m], yr[m], sv[mv], stv[mv], yrv[mv])
        return self

    def refit(self, hourly, rows, issue, level, y, epochs: int):
        self._prepare_base(hourly.loc[: rows.max()])
        s, st, ok = self._arrays(hourly, rows, issue, level)
        yr = y / level
        m = ok & np.isfinite(yr)
        self.net_, _, _ = self._train(s[m], st[m], yr[m], epochs=max(1, epochs))
        return self

    def predict(self, hourly, rows, issue, level) -> np.ndarray:
        s, st, ok = self._arrays(hourly, rows, issue, level)
        p = self._predict_raw(self.net_, s, st) * level
        p[~ok] = np.nan
        return p

    def __getstate__(self):
        d = self.__dict__.copy()
        if "net_" in d:
            d["net_state_"] = d.pop("net_").state_dict()
            d["net_shape_"] = (len(INPUT_SERIES) + 3, 8)
        return d

    def __setstate__(self, d):
        state = d.pop("net_state_", None)
        shape = d.pop("net_shape_", None)
        self.__dict__.update(d)
        if state is not None:
            self.net_ = _Net(shape[0], shape[1], self.hidden, self.layers, self.dropout)
            self.net_.load_state_dict(state)
