"""Surplus-solar risk PROXY (scenario indicator, not observed curtailment).

    F_t = theta * ThermalCommitted_d + Nuclear_t + HydroMin_d
    S_t(k) = max(0, k * Solar_t + Wind_t + F_t - Demand_t)            [MW]

ThermalCommitted_d = max hourly thermal output on day d (proxy for capacity online);
theta = technical-minimum share (CEA 2023 regulations: 0.55 interim, 0.40 target);
HydroMin_d = min hourly hydro on day d (proxy for must-run hydro);
k = solar scale-up factor (1 = observed fleet). Only defined where the generation mix is
available (2021-09 .. 2023-12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THETAS = (0.55, 0.40)
SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
MIX_COLS = ["demand_mw", "solar_mw", "wind_mw", "thermal_mw", "hydro_mw", "nuclear_mw"]


def inflexible_floor(hourly: pd.DataFrame, theta: float) -> pd.Series:
    day = hourly.index.normalize()
    thermal_committed = hourly["thermal_mw"].groupby(day).transform("max")
    hydro_min = hourly["hydro_mw"].groupby(day).transform("min")
    return theta * thermal_committed + hourly["nuclear_mw"] + hydro_min


def surplus_proxy(hourly: pd.DataFrame, theta: float, k: float) -> pd.Series:
    d = hourly.dropna(subset=MIX_COLS)
    f = inflexible_floor(d, theta)
    s = (k * d["solar_mw"] + d["wind_mw"] + f - d["demand_mw"]).clip(lower=0)
    return s.rename(f"surplus_mw_theta{theta}_k{k}")


def headroom(hourly: pd.DataFrame, theta: float, k: float = 1.0) -> pd.Series:
    """Signed margin Demand - (k*Solar + Wind + F) [MW]; negative = surplus under the proxy."""
    d = hourly.dropna(subset=MIX_COLS)
    return d["demand_mw"] - (k * d["solar_mw"] + d["wind_mw"] + inflexible_floor(d, theta))


def scenario_table(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theta in THETAS:
        for k in SCALES:
            s = surplus_proxy(hourly, theta, k)
            days = s.groupby(s.index.normalize()).sum()
            rows.append({
                "theta": theta, "solar_scale_k": k, "hours": len(s),
                "surplus_hours": int((s > 0).sum()),
                "surplus_hours_pct": 100 * float((s > 0).mean()),
                "days_with_surplus": int((days > 0).sum()),
                "surplus_energy_gwh": float(s.sum() / 1e3),
                "mean_surplus_mwh_per_day": float(days.mean()),
                "max_surplus_mw": float(s.max()),
                "solar_energy_scaled_gwh": float(k * hourly.loc[s.index, "solar_mw"].sum() / 1e3),
            })
    out = pd.DataFrame(rows)
    out["surplus_share_of_solar_pct"] = 100 * out["surplus_energy_gwh"] / out["solar_energy_scaled_gwh"]
    return out


def thermal_sanity(hourly: pd.DataFrame, theta: float) -> float:
    """Fraction of hours in which observed thermal output is below theta * daily max (should be small)."""
    d = hourly.dropna(subset=["thermal_mw"])
    tc = d["thermal_mw"].groupby(d.index.normalize()).transform("max")
    return float(np.mean(d["thermal_mw"] < theta * tc))
