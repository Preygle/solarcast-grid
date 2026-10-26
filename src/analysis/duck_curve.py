"""Duck-curve metrics from the hourly table (DERIVED from OBSERVED data).

Net load = demand - solar - wind (variable renewables; hydro treated as dispatchable).
Evening ramp (day d) = max over start hours s in [15, 18] of NL(d, s+3) - NL(d, s), i.e. the
largest 3-hour increase that starts between 15:00 and 18:00 IST, in MW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASONS = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
           3: "Pre-monsoon (MAM)", 4: "Pre-monsoon (MAM)", 5: "Pre-monsoon (MAM)",
           6: "Monsoon (JJAS)", 7: "Monsoon (JJAS)", 8: "Monsoon (JJAS)", 9: "Monsoon (JJAS)",
           10: "Post-monsoon (ON)", 11: "Post-monsoon (ON)"}
RAMP_STARTS = [15, 16, 17, 18]
RAMP_LEN_H = 3


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = out.index.month.map(SEASONS)
    out["year"] = out.index.year
    out["hour"] = out.index.hour
    return out


def daily_pivot(s: pd.Series) -> pd.DataFrame:
    """Day x hour matrix (complete days only)."""
    df = s.to_frame("v")
    df["date"], df["hour"] = df.index.normalize(), df.index.hour
    p = df.pivot_table(index="date", columns="hour", values="v", aggfunc="first")
    return p.dropna()


def daily_duck_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-day net-load statistics and evening ramp."""
    nl = daily_pivot(hourly["net_load_mw"])
    dem = daily_pivot(hourly["demand_mw"]).reindex(nl.index)
    sol = daily_pivot(hourly["solar_mw"]).reindex(nl.index)
    ramps = np.column_stack([nl[s + RAMP_LEN_H] - nl[s] for s in RAMP_STARTS])
    out = pd.DataFrame({
        "net_load_min_mw": nl.min(axis=1),
        "net_load_min_hour": nl.idxmin(axis=1),
        "net_load_max_mw": nl.max(axis=1),
        "net_load_max_hour": nl.idxmax(axis=1),
        "net_load_range_mw": nl.max(axis=1) - nl.min(axis=1),
        "evening_ramp_3h_mw": ramps.max(axis=1),
        "evening_ramp_start_hour": np.array(RAMP_STARTS)[ramps.argmax(axis=1)],
        "demand_max_mw": dem.max(axis=1),
        "solar_energy_mwh": sol.sum(axis=1),
        "demand_energy_mwh": dem.sum(axis=1),
    }, index=nl.index)
    out["solar_share_pct"] = 100 * out["solar_energy_mwh"] / out["demand_energy_mwh"]
    # "Duck day": midday (11-14h) net load trough below both morning (6-9h) and evening (18-21h) levels.
    mid = nl[[11, 12, 13, 14]].mean(axis=1)
    out["duck_shape"] = (mid < nl[[6, 7, 8, 9]].mean(axis=1)) & (mid < nl[[18, 19, 20, 21]].mean(axis=1))
    return out


def seasonal_profiles(hourly: pd.DataFrame, col: str = "net_load_mw") -> pd.DataFrame:
    """Mean hourly profile by (year, season)."""
    d = add_season(hourly)
    return d.groupby(["year", "season", "hour"])[col].mean().unstack("hour")
