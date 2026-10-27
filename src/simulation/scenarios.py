"""Map all-India profiles onto the benchmark feeder, and generate labelled power-flow scenarios.

Profile mapping (a documented scenario device, not a claim about any real feeder):
    load_scale_t = demand_t / D_REF           (D_REF = max hourly demand in train+val, MW)
    pv_pu_t      = min(1, solar_t / S_ref_d)  (S_ref_d = rolling 30-day max solar known at issue time)
Feeder load = case33bw base load (3.715 MW peak) x load_scale_t; PV per site = PV_RATED_MW x pv_pu_t.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.grid import network as gn
from src.power_flow.runner import OperatingPoint, run

TRAINVAL_END = "2024-12-31 23:00"


def reference_levels(hourly: pd.DataFrame) -> dict:
    d_ref = float(hourly.loc[:TRAINVAL_END, "demand_mw"].max())
    s_ref = hourly["solar_mw"].rolling(720, min_periods=240).max().shift(1)
    return {"d_ref": d_ref, "s_ref": s_ref}


def map_day(demand: pd.Series, solar: pd.Series, refs: dict) -> pd.DataFrame:
    """Hourly feeder profile for the given (observed or forecast) all-India series. The solar
    reference is the value known at the day-ahead issue time (D-1 09:00), identical for forecast
    and actual profiles of the same day."""
    issue = solar.index.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=9)
    sref = refs["s_ref"].reindex(issue).ffill().bfill()
    sref.index = solar.index
    return pd.DataFrame({
        "load_scale": (demand / refs["d_ref"]).to_numpy(),
        "pv_pu": np.clip(solar / sref, 0, 1).to_numpy(),
    }, index=demand.index)


# ---------------- receiving-end / loss scenario generator ----------------

def generate(hourly: pd.DataFrame, configs: list[list[int]], n_families: int, seed: int = 0,
             battery_bus: int | None = None, log=print) -> pd.DataFrame:
    """Each scenario family = one random observed day x one radial configuration x random PV
    capacities, battery power profile and per-bus load noise; 24 hourly operating points."""
    rng = np.random.default_rng(seed)
    refs = reference_levels(hourly)
    days = hourly.dropna(subset=["demand_mw", "solar_mw"]).index.normalize().unique()
    days = days[days >= pd.Timestamp("2021-10-01")]
    net = gn.base_network()
    if battery_bus is not None:
        gn.set_battery_bus(net, battery_bus)
    rows = []
    t0 = time.time()
    for fam in range(n_families):
        day = days[rng.integers(len(days))]
        h = hourly.loc[day: day + pd.Timedelta(hours=23)]
        if len(h) < 24 or h[["demand_mw", "solar_mw"]].isna().any().any():
            continue
        prof = map_day(h["demand_mw"], h["solar_mw"], refs)
        cfg_id = int(rng.integers(len(configs)))
        gn.set_configuration(net, configs[cfg_id])
        pv_cap = gn.PV_RATED_MW * rng.uniform(0.0, 1.5, size=len(gn.PV_BUSES))
        load_mult = rng.uniform(0.7, 1.3)
        bus_noise = rng.lognormal(0.0, 0.10, size=len(net.load))
        batt_amp = rng.uniform(0, 2.0)
        batt_phase = rng.uniform(0, 2 * np.pi)
        for k, (ts, r) in enumerate(prof.iterrows()):
            scale = r.load_scale * load_mult * bus_noise
            pv = pv_cap * r.pv_pu
            batt = batt_amp * np.sin(2 * np.pi * k / 24 + batt_phase)
            op = OperatingPoint(load_scale=scale, pv_mw=pv, battery_mw=batt)
            res = run(net, op)
            p_load = net.load.p_mw.to_numpy()
            q_load = net.load.q_mvar.to_numpy()
            row = {"family": fam, "seed": seed, "source_day": day.date().isoformat(), "hour": ts.hour,
                   "config_id": cfg_id, "open_lines": "-".join(map(str, configs[cfg_id])),
                   "label_source": "SIMULATED (pandapower AC PF, case33bw)",
                   "load_p_total_mw": p_load.sum(), "load_q_total_mvar": q_load.sum(),
                   "battery_mw": batt, **{f"pv_{b}_mw": v for b, v in zip(gn.PV_BUSES, pv)}}
            # Zonal loads (lateral groups) as inputs; they are known operating conditions, not outputs.
            for zone, buses in ZONES.items():
                row[f"load_{zone}_mw"] = float(net.load.loc[net.load.bus.isin(buses), "p_mw"].sum())
            row.update(res)
            rows.append(row)
        if fam % 100 == 99:
            log(f"   {fam + 1}/{n_families} families ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


ZONES = {  # contiguous bus groups of the default radial feeder
    "main_1_5": list(range(1, 6)), "main_6_17": list(range(6, 18)), "lat_18_21": list(range(18, 22)),
    "lat_22_24": list(range(22, 25)), "lat_25_32": list(range(25, 33)),
}
