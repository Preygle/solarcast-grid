"""AC power flow for one operating point, with loss / receiving-end accounting and checks.

Sign conventions (all MW):
* ``p_recv_mw``  = power delivered INTO the upstream grid at the receiving bus (bus 0)
                 = -P(ext_grid). Negative means the feeder imports.
* ``p_sent_mw``  = all power injected into the feeder: PV + battery discharge + import.
* ``p_loss_mw``  = sum over in-service lines of (p_from + p_to) = technical (I^2 R) losses.
* ``p_delivered_mw`` = power reaching a sink = loads + battery charging + export = p_sent - p_loss.
* ``loss_pct``   = 100 * p_loss / p_sent.
Energy balance check: sum(PV) + P_batt - sum(load) - p_loss - p_recv = 0 (|residual| < 1e-6 MW).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandapower as pp

from src.grid.network import RECEIVING_BUS, V_MAX, V_MIN

BALANCE_TOL_MW = 1e-6


@dataclass
class OperatingPoint:
    load_scale: np.ndarray | float      # per-load multiplier on base P/Q (scalar or len 32)
    pv_mw: np.ndarray                   # injection per PV site (after curtailment)
    battery_mw: float = 0.0             # + discharge (injection), - charge
    flex_mw: np.ndarray | None = None   # per-load additive change in P (demand shifting)


def apply(net, op: OperatingPoint) -> None:
    scale = np.broadcast_to(np.asarray(op.load_scale, dtype=float), (len(net.load),))
    net.load["p_mw"] = net.load["p_base_mw"].to_numpy() * scale
    net.load["q_mvar"] = net.load["q_base_mvar"].to_numpy() * scale
    if op.flex_mw is not None:
        pf = net.load["q_base_mvar"].to_numpy() / net.load["p_base_mw"].to_numpy()
        net.load["p_mw"] = net.load["p_mw"].to_numpy() + op.flex_mw
        net.load["q_mvar"] = net.load["q_mvar"].to_numpy() + op.flex_mw * pf
    pv = net.sgen.type == "PV"
    net.sgen.loc[pv, "p_mw"] = np.asarray(op.pv_mw, dtype=float)
    net.sgen.loc[net.sgen.name == "battery", "p_mw"] = float(op.battery_mw)


def run(net, op: OperatingPoint | None = None) -> dict:
    if op is not None:
        apply(net, op)
    try:
        pp.runpp(net, algorithm="nr", max_iteration=30, numba=True)
        converged = bool(net.converged)
    except pp.LoadflowNotConverged:
        converged = False
    out = {"converged": converged}
    if not converged:
        return out
    lines = net.res_line[net.line.in_service]
    p_loss = float((lines.p_from_mw + lines.p_to_mw).sum())
    p_ext = float(net.res_ext_grid.p_mw.sum())
    pv = float(net.sgen.loc[net.sgen.type == "PV", "p_mw"].sum())
    batt = float(net.sgen.loc[net.sgen.name == "battery", "p_mw"].sum())
    load = float(net.load.p_mw.sum())
    p_recv = -p_ext
    p_sent = pv + max(batt, 0.0) + max(p_ext, 0.0)
    residual = pv + batt - load - p_loss - p_recv
    vm = net.res_bus.vm_pu
    out.update({
        "p_loss_mw": p_loss, "p_recv_mw": p_recv, "p_import_mw": max(p_ext, 0.0), "p_export_mw": max(-p_ext, 0.0),
        "p_sent_mw": p_sent, "p_delivered_mw": p_sent - p_loss,
        "loss_pct": 100 * p_loss / p_sent if p_sent > 0 else np.nan,
        "p_load_mw": load, "p_pv_mw": pv, "p_batt_mw": batt,
        "q_ext_mvar": float(net.res_ext_grid.q_mvar.sum()),
        "vm_min_pu": float(vm.min()), "vm_max_pu": float(vm.max()),
        "vm_min_bus": int(vm.idxmin()), "vm_max_bus": int(vm.idxmax()),
        "line_loading_max_pct": float(lines.loading_percent.max()),
        "line_loading_max_id": int(lines.loading_percent.idxmax()),
        "v_violation": bool(vm.min() < V_MIN - 1e-6 or vm.max() > V_MAX + 1e-6),
        "thermal_violation": bool(lines.loading_percent.max() > 100 + 1e-6),
        "balance_residual_mw": residual,
    })
    if abs(residual) > BALANCE_TOL_MW:
        raise AssertionError(f"energy balance residual {residual:.3e} MW at receiving bus {RECEIVING_BUS}")
    return out
