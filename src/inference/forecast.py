"""Prediction interface for downstream components (optimisation, dashboard).

Training logic lives in ``src/training``; this module only reloads saved artefacts listed in
``models/registry.json`` and produces forecasts from history available at the issue time.

    fc = Forecaster()
    df = fc.day_ahead(hourly, "2025-04-12")     # 24 rows: solar/demand/wind/net-load + 80% PI
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import FeatureSpec, build_features
from src.models.lstm import LSTMForecaster
from src.models.tabular import LEVEL_COL
from src.training.train import postprocess

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "models" / "registry.json"


class ModelUnavailable(RuntimeError):
    pass


class Forecaster:
    def __init__(self, registry_path: Path = REGISTRY):
        if not registry_path.exists():
            raise ModelUnavailable(f"{registry_path} not found; run scripts/train_models.py")
        self.registry = json.loads(registry_path.read_text())
        self._cache: dict = {}

    def _load(self, task: str) -> dict:
        if task not in self._cache:
            entry = self.registry.get(task)
            if entry is None or entry.get("artifact") is None:
                raise ModelUnavailable(f"no saved ML model for task {task!r}")
            path = ROOT / entry["artifact"]
            if not path.exists():
                raise ModelUnavailable(f"artefact missing: {path}")
            self._cache[task] = joblib.load(path)
        return self._cache[task]

    def predict(self, hourly: pd.DataFrame, target: str, horizon: str,
                when: pd.DatetimeIndex) -> pd.DataFrame:
        """Forecasts for target hours ``when``. Values after each row's issue time are never used:
        the history is truncated at the earliest issue time and features are computed from it."""
        task = f"{target.replace('_mw', '')}_{horizon}"
        art = self._load(task)
        spec = FeatureSpec(**art["spec"])
        from src.features.build import issue_time
        tau = issue_time(when, horizon)
        hist = hourly.loc[: tau.max()].copy()
        # Extend the index to the target hours (unknown values = NaN) so features can be aligned.
        full_idx = hist.index.union(when)
        hist = hist.reindex(full_idx)
        hist.loc[hist.index > tau.max(), :] = np.nan
        hist[target] = hist[target].fillna(-1.0)  # placeholder so rows survive the y.notna() filter
        X, _, meta = build_features(hist, spec)
        X = X.loc[when, art["features"]]
        model = art["model"]
        if isinstance(model, LSTMForecaster):
            lv = X[LEVEL_COL[target]].to_numpy(dtype=float)
            lv = np.where(np.isfinite(lv) & (lv > 0), lv, np.nanmedian(lv))
            hist2 = hourly.loc[: tau.max()].reindex(full_idx)
            p = model.predict(hist2, when, meta.loc[when, "issue_time"], lv)
        else:
            p = model.predict(X)
        p = postprocess(target, when, p)
        hw = self.registry[task]["pi_halfwidth_by_hour"]
        half = np.array([hw[str(hr)] if str(hr) in hw else hw[hr] for hr in when.hour])
        lo = p - half
        if target in ("solar_mw", "wind_mw"):
            lo = np.clip(lo, 0, None)
        return pd.DataFrame({target: p, f"{target}_pi_lo": lo, f"{target}_pi_hi": p + half,
                             "issue_time": meta.loc[when, "issue_time"].to_numpy(),
                             "model": self.registry[task]["selected_model"]}, index=when)

    def day_ahead(self, hourly: pd.DataFrame, date) -> pd.DataFrame:
        when = pd.date_range(pd.Timestamp(date), periods=24, freq="h")
        parts = [self.predict(hourly, t, "da", when) for t in ["solar_mw", "demand_mw", "wind_mw"]]
        out = pd.concat([p.drop(columns=["issue_time", "model"]) for p in parts], axis=1)
        out["net_load_mw"] = out["demand_mw"] - out["solar_mw"] - out["wind_mw"]
        out["issue_time"] = parts[0]["issue_time"]
        out["models"] = "/".join(p["model"].iloc[0] for p in parts)
        return out
