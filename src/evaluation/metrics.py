"""Forecast error metrics. All absolute metrics are in the target's unit (MW)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# MAPE is undefined/degenerate near zero (Hyndman & Koehler 2006), so it is reported
# only for targets that stay far from zero.
MAPE_OK = {"demand_mw", "net_load_mw"}


def regression_metrics(y: pd.Series, p: pd.Series, target: str, mase_scale: float | None = None) -> dict:
    m = pd.concat([y.rename("y"), pd.Series(p, index=y.index).rename("p")], axis=1).dropna()
    e = m["p"] - m["y"]
    out = {
        "n": int(len(m)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "bias": float(e.mean()),
        "r2": float(1 - (e ** 2).sum() / ((m["y"] - m["y"].mean()) ** 2).sum()),
        "nmae_pct": float(100 * e.abs().mean() / m["y"].abs().mean()),
    }
    if target in MAPE_OK:
        out["mape_pct"] = float(100 * (e.abs() / m["y"].abs()).mean())
    if target == "solar_mw":
        day = m["y"] > 0.05 * m["y"].max()
        out["mae_daylight"] = float(e[day].abs().mean())
        out["nmae_daylight_pct"] = float(100 * e[day].abs().mean() / m.loc[day, "y"].mean())
    if mase_scale:
        out["mase"] = float(e.abs().mean() / mase_scale)
    return out


def mase_scale(y_train: pd.Series, season: int = 24) -> float:
    """In-sample MAE of the seasonal-naive (24 h) forecast on the training set."""
    return float((y_train - y_train.shift(season)).abs().mean())
