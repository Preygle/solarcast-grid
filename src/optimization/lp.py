"""Day-ahead (24 h) LP for solar utilisation on the radial feeder (PuLP + HiGHS).

Decisions per hour t:  PV used per site u_kt <= available a_kt (curtailment c_kt = a_kt - u_kt),
battery charge ch_t / discharge dis_t, state of charge soc_t, flexible-load shift d_t
as a total feeder shift d_t [MW] spread over loads in proportion to base load (energy-neutral
over the day, |d_t| <= flex_frac x planned load).

Network model: LinDistFlow on the radial configuration (active/reactive flows, squared voltages),
no losses in the balance (standard LinDistFlow); losses enter the OBJECTIVE through a convex
piecewise-linear outer approximation of r * P^2 (tangent cuts).

Objective (MWh-equivalent):
    min  W_CURT * sum c  +  W_LOSS * sum loss_proxy  +  W_RAMP * sum |head_t - head_{t-1}|
         +  W_DEG * sum (ch + dis)  +  M * slack
Curtailed energy and loss energy are both lost energy, so they share weight 1 by default.
Constraints: SoC dynamics with efficiency, SoC bounds, cyclic SoC (soc_24 >= soc_0), power limits,
flex bounds and energy neutrality, export limit at the receiving bus, line active-power limits,
voltage limits with a safety margin. Soft slacks (penalty M) are only active in the
*realisation* stage where decisions fixed from a forecast may meet different actual conditions;
any slack use is reported as a constraint violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pulp

from src.grid import network as gn
from src.grid.graph import radial_tree

S_BASE_MVA = 1.0
V_BASE_KV = 12.66
Z_BASE = V_BASE_KV ** 2 / S_BASE_MVA
TANGENTS_MW = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
V_MARGIN = 0.005
FLOW_MARGIN = 0.9


@dataclass
class BatteryParams:
    bus: int = gn.BATTERY_BUS
    power_mw: float = 1.5
    energy_mwh: float = 6.0
    eta: float = 0.95            # one-way efficiency (round trip 0.9025)
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_init: float = 0.5


@dataclass
class Weights:
    curt: float = 1.0
    loss: float = 1.0
    ramp: float = 0.05
    deg: float = 0.005
    slack: float = 1e3


@dataclass
class DayInputs:
    load_scale: np.ndarray            # (24,)
    pv_avail_mw: np.ndarray           # (24, n_pv)
    flex_frac: float = 0.10           # |f_t| <= flex_frac
    export_limit_mw: float = gn.EXPORT_LIMIT_MW


@dataclass
class Fixed:
    """Decisions fixed from an earlier (forecast-based) plan."""
    ch: np.ndarray | None = None
    dis: np.ndarray | None = None
    flex: np.ndarray | None = None


@dataclass
class LPResult:
    status: str
    objective: float
    schedule: dict = field(default_factory=dict)
    terms: dict = field(default_factory=dict)


def _var(name, lo=None, hi=None):
    return pulp.LpVariable(name, lowBound=lo, upBound=hi)


def _tree_data(net):
    t = radial_tree(net)
    parent = dict(nx.bfs_predecessors(t, gn.RECEIVING_BUS))
    order = list(nx.bfs_tree(t, gn.RECEIVING_BUS).nodes)
    lines = {}
    for j, i in parent.items():
        d = t.edges[i, j]
        lines[j] = {"from": i, "line": d["line"], "r": d["r_ohm"] / Z_BASE, "x": d["x_ohm"] / Z_BASE,
                    "pmax": FLOW_MARGIN * np.sqrt(3) * V_BASE_KV * d["max_i_ka"]}
    children = {}
    for j, i in parent.items():
        children.setdefault(i, []).append(j)
    return parent, order, lines, children


def solve_day(net, inp: DayInputs, batt: BatteryParams | None, w: Weights = Weights(),
              fixed: Fixed | None = None, msg: bool = False) -> LPResult:
    T = len(inp.load_scale)
    parent, order, lines, children = _tree_data(net)
    buses = order
    p_base = net.load.groupby("bus")["p_base_mw"].sum().reindex(buses, fill_value=0.0).to_dict()
    q_base = net.load.groupby("bus")["q_base_mvar"].sum().reindex(buses, fill_value=0.0).to_dict()
    pv_bus = gn.PV_BUSES
    has_batt = batt is not None and batt.power_mw > 0
    soft = fixed is not None

    m = pulp.LpProblem("day", pulp.LpMinimize)
    u = {(k, t): _var(f"u_{k}_{t}", 0, float(inp.pv_avail_mw[t, k])) for k in range(len(pv_bus)) for t in range(T)}
    total_base = sum(p_base.values())
    use_flex = inp.flex_frac > 0
    # Flexible load = total feeder shift d_t [MW], spread over loads in proportion to base load.
    # Planning bounds |d_t| <= flex_frac * planned load; when d is fixed from a plan, it is applied
    # as scheduled MW (energy-neutral by construction) regardless of the actual load.
    f = {t: _var(f"d_{t}") if use_flex else 0 for t in range(T)}
    if has_batt:
        ch = {t: _var(f"ch_{t}", 0, batt.power_mw) for t in range(T)}
        dis = {t: _var(f"dis_{t}", 0, batt.power_mw) for t in range(T)}
        soc = {t: _var(f"soc_{t}", batt.soc_min * batt.energy_mwh, batt.soc_max * batt.energy_mwh) for t in range(T)}
    else:
        ch = dis = {t: 0 for t in range(T)}
    P = {(j, t): _var(f"P_{j}_{t}") for j in lines for t in range(T)}
    Q = {(j, t): _var(f"Q_{j}_{t}") for j in lines for t in range(T)}
    v = {(b, t): _var(f"v_{b}_{t}") for b in buses for t in range(T)}
    loss = {(j, t): _var(f"L_{j}_{t}", 0) for j in lines for t in range(T)}
    head = {t: _var(f"head_{t}") for t in range(T)}
    ramp = {t: _var(f"ramp_{t}", 0) for t in range(1, T)}
    sv = {(b, t): _var(f"sv_{b}_{t}", 0) for b in buses for t in range(T)} if soft else {}
    sp = {(j, t): _var(f"sp_{j}_{t}", 0) for j in lines for t in range(T)} if soft else {}
    se = {t: _var(f"se_{t}", 0) for t in range(T)} if soft else {}

    if fixed is not None:
        for t in range(T):
            if has_batt and fixed.ch is not None:
                m += ch[t] == float(fixed.ch[t])
                m += dis[t] == float(fixed.dis[t])
            if use_flex and fixed.flex is not None:
                m += f[t] == float(fixed.flex[t])
    if use_flex and (fixed is None or fixed.flex is None):
        for t in range(T):
            m += f[t] <= inp.flex_frac * inp.load_scale[t] * total_base
            m += -f[t] <= inp.flex_frac * inp.load_scale[t] * total_base
    for t in range(T):
        # net withdrawal at each bus
        pinj = {b: 0 for b in buses}
        qw = {b: 0 for b in buses}
        for b in buses:
            share = p_base[b] / total_base
            pinj[b] = p_base[b] * inp.load_scale[t] + (f[t] * share if use_flex else 0)
            pf = q_base[b] / p_base[b] if p_base[b] > 0 else 0.0
            qw[b] = q_base[b] * inp.load_scale[t] + (f[t] * share * pf if use_flex else 0)
        for k, b in enumerate(pv_bus):
            pinj[b] = pinj[b] - u[k, t]
        if has_batt:
            pinj[batt.bus] = pinj[batt.bus] + ch[t] - dis[t]
        # flow conservation: P_j = withdrawal_j + sum children P
        for j in lines:
            m += P[j, t] == pinj[j] + pulp.lpSum(P[c, t] for c in children.get(j, []))
            m += Q[j, t] == qw[j] + pulp.lpSum(Q[c, t] for c in children.get(j, []))
            i = lines[j]["from"]
            m += v[j, t] == v[i, t] - 2 * (lines[j]["r"] * P[j, t] + lines[j]["x"] * Q[j, t])
            pmax = lines[j]["pmax"]
            m += P[j, t] <= pmax + (sp[j, t] if soft else 0)
            m += -P[j, t] <= pmax + (sp[j, t] if soft else 0)
            for p0 in TANGENTS_MW:
                m += loss[j, t] >= lines[j]["r"] * (2 * p0 * P[j, t] - p0 ** 2)
                m += loss[j, t] >= lines[j]["r"] * (-2 * p0 * P[j, t] - p0 ** 2)
        m += v[gn.RECEIVING_BUS, t] == 1.0
        for b in buses:
            if b == gn.RECEIVING_BUS:
                continue
            m += v[b, t] <= (gn.V_MAX - V_MARGIN) ** 2 + (sv[b, t] if soft else 0)
            m += v[b, t] >= (gn.V_MIN + V_MARGIN) ** 2 - (sv[b, t] if soft else 0)
        m += head[t] == pulp.lpSum(P[c, t] for c in children[gn.RECEIVING_BUS])
        m += -head[t] <= inp.export_limit_mw + (se[t] if soft else 0)
        if t > 0:
            m += ramp[t] >= head[t] - head[t - 1]
            m += ramp[t] >= head[t - 1] - head[t]
        if has_batt:
            prev = batt.soc_init * batt.energy_mwh if t == 0 else soc[t - 1]
            m += soc[t] == prev + batt.eta * ch[t] - dis[t] * (1 / batt.eta)
    if has_batt:
        m += soc[T - 1] >= batt.soc_init * batt.energy_mwh
    if use_flex and (fixed is None or fixed.flex is None):
        m += pulp.lpSum(f[t] for t in range(T)) == 0  # energy-neutral shifting over the day

    curt = pulp.lpSum(float(inp.pv_avail_mw[t, k]) - u[k, t] for k in range(len(pv_bus)) for t in range(T))
    loss_e = pulp.lpSum(loss.values())
    ramp_s = pulp.lpSum(ramp.values())
    thr = pulp.lpSum(ch[t] + dis[t] for t in range(T)) if has_batt else 0
    slack = pulp.lpSum(sv.values()) + pulp.lpSum(sp.values()) + pulp.lpSum(se.values()) if soft else 0
    m += w.curt * curt + w.loss * loss_e + w.ramp * ramp_s + w.deg * thr + w.slack * slack

    m.solve(pulp.HiGHS(msg=msg))
    status = pulp.LpStatus[m.status]
    if status != "Optimal":
        return LPResult(status, float("nan"))
    val = lambda e: float(pulp.value(e)) if not isinstance(e, (int, float)) else float(e)
    sched = {
        "pv_used_mw": np.array([[val(u[k, t]) for k in range(len(pv_bus))] for t in range(T)]),
        "ch_mw": np.array([val(ch[t]) for t in range(T)]),
        "dis_mw": np.array([val(dis[t]) for t in range(T)]),
        "soc_mwh": np.array([val(soc[t]) for t in range(T)]) if has_batt else np.zeros(T),
        "flex_mw": np.array([val(f[t]) for t in range(T)]),
        "head_mw": np.array([val(head[t]) for t in range(T)]),
        "v_lin_max_pu": np.sqrt(max(val(v[b, t]) for b in buses for t in range(T))),
        "v_lin_min_pu": np.sqrt(min(val(v[b, t]) for b in buses for t in range(T))),
    }
    terms = {"curtailment_mwh": val(curt), "loss_proxy_mwh": val(loss_e), "ramp_sum_mw": val(ramp_s),
             "throughput_mwh": val(thr), "slack_total": val(slack) if soft else 0.0,
             "objective": float(pulp.value(m.objective))}
    return LPResult(status, terms["objective"], sched, terms)
