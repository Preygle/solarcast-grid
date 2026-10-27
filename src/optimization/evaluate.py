"""Baseline vs optimised operation on identical days, with AC power-flow validation.

Cases (same feeder, same actual day; PV / load mapped from OBSERVED all-India data):
* ``baseline``           - no battery, no flexible load; PV curtailed only as much as the
                           export / voltage / thermal limits require (curtailment-only LP on actuals).
* ``rule_based``         - battery charges from would-be-curtailed surplus, discharges evenly 17-22 h.
* ``forecast_informed``  - battery + flex schedule from the LP on DAY-AHEAD FORECASTS (Phase 3
                           models), then realised on the actual day with those decisions fixed and
                           curtailment re-optimised (soft-constraint slacks reported as violations).
* ``perfect_foresight``  - the same LP on actual profiles (upper bound on achievable benefit).
Every realised hour is re-run with full AC power flow; reported losses, voltages, loadings and
receiving-end energy come from the AC results, not from the LP approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.grid import network as gn
from src.optimization.lp import BatteryParams, DayInputs, Fixed, Weights, solve_day
from src.power_flow.runner import OperatingPoint, run

CASES = ["baseline", "rule_based", "forecast_informed", "perfect_foresight"]
EVENING = range(16, 22)


def rule_based_schedule(inp: DayInputs, batt: BatteryParams, base_load_mw: float) -> Fixed:
    T = len(inp.load_scale)
    ch, dis = np.zeros(T), np.zeros(T)
    e_max, e_min = batt.soc_max * batt.energy_mwh, batt.soc_min * batt.energy_mwh
    soc = batt.soc_init * batt.energy_mwh
    for t in range(T):
        surplus = inp.pv_avail_mw[t].sum() - inp.load_scale[t] * base_load_mw - inp.export_limit_mw
        if surplus > 0:
            ch[t] = min(batt.power_mw, surplus, (e_max - soc) / batt.eta)
            soc += batt.eta * ch[t]
    target_end = batt.soc_init * batt.energy_mwh
    ev = [t for t in range(T) if t in EVENING]
    avail = max(0.0, soc - max(target_end, e_min))
    for t in ev:
        d = min(batt.power_mw, avail * batt.eta / len(ev))
        dis[t] = d
    return Fixed(ch=ch, dis=dis, flex=np.zeros(T))


def ac_validate(net, actual: DayInputs, sched: dict) -> dict:
    T = len(actual.load_scale)
    recs = []
    for t in range(T):
        share = net.load["p_base_mw"].to_numpy() / net.load["p_base_mw"].sum()
        flex = sched["flex_mw"][t] * share
        op = OperatingPoint(actual.load_scale[t], sched["pv_used_mw"][t], sched["dis_mw"][t] - sched["ch_mw"][t], flex)
        r = run(net, op)
        if not r["converged"]:
            recs.append({"hour": t, "converged": False})
            continue
        recs.append({"hour": t, **r})
    df = pd.DataFrame(recs)
    head = -df["p_recv_mw"]  # feeder net load seen from the upstream grid (import positive)
    ramps = [head.iloc[s + 3] - head.iloc[s] for s in range(15, 19)]
    exp_viol = (df["p_export_mw"] > actual.export_limit_mw + 1e-3)
    return {
        "ac_converged_hours": int(df["converged"].sum()),
        "loss_mwh": float(df["p_loss_mw"].sum()),
        "received_export_mwh": float(df["p_export_mw"].sum()),
        "import_mwh": float(df["p_import_mw"].sum()),
        "net_received_mwh": float(df["p_recv_mw"].sum()),
        "delivered_mwh": float(df["p_delivered_mw"].sum()),
        "sent_mwh": float(df["p_sent_mw"].sum()),
        "peak_import_mw": float(head.max()),
        "evening_ramp_3h_mw": float(max(ramps)),
        "vm_max_pu": float(df["vm_max_pu"].max()), "vm_min_pu": float(df["vm_min_pu"].min()),
        "line_loading_max_pct": float(df["line_loading_max_pct"].max()),
        "v_violation_hours": int(df["v_violation"].sum()),
        "thermal_violation_hours": int(df["thermal_violation"].sum()),
        "export_violation_hours": int(exp_viol.sum()),
        "hourly": df,
    }


def run_day(net, actual: DayInputs, forecast: DayInputs, batt: BatteryParams, w: Weights = Weights(),
            cases=CASES) -> dict:
    out = {}
    avail = float(actual.pv_avail_mw.sum())
    base_load = float(net.load["p_base_mw"].sum())
    no_flex = DayInputs(actual.load_scale, actual.pv_avail_mw, flex_frac=0.0, export_limit_mw=actual.export_limit_mw)
    for case in cases:
        if case == "baseline":
            res = solve_day(net, no_flex, None, w)
            plan_status = res.status
        elif case == "rule_based":
            fx = rule_based_schedule(actual, batt, base_load)
            res = solve_day(net, no_flex, batt, w, fixed=fx)
            plan_status = "rule"
        elif case == "forecast_informed":
            plan = solve_day(net, forecast, batt, w)
            plan_status = plan.status
            if plan.status != "Optimal":
                out[case] = {"plan_status": plan.status}
                continue
            fx = Fixed(ch=plan.schedule["ch_mw"], dis=plan.schedule["dis_mw"], flex=plan.schedule["flex_mw"])
            res = solve_day(net, actual, batt, w, fixed=fx)
        elif case == "perfect_foresight":
            res = solve_day(net, actual, batt, w)
            plan_status = res.status
        else:
            raise ValueError(case)
        if res.status != "Optimal":
            out[case] = {"plan_status": plan_status, "realise_status": res.status}
            continue
        ac = ac_validate(net, actual, res.schedule)
        out[case] = {"plan_status": plan_status, "realise_status": res.status,
                     "pv_available_mwh": avail, "curtailment_mwh": res.terms["curtailment_mwh"],
                     "pv_used_mwh": avail - res.terms["curtailment_mwh"],
                     "battery_throughput_mwh": res.terms["throughput_mwh"],
                     "lp_slack_total": res.terms["slack_total"], "lp_loss_proxy_mwh": res.terms["loss_proxy_mwh"],
                     "schedule": res.schedule, **ac}
    return out
