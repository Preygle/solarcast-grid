"""Benchmark distribution feeder: Baran & Wu 33-bus, 12.66 kV (pandapower ``case33bw``).

Everything here is SIMULATED. Additions to the published case are explicit ASSUMPTIONS:
* Line thermal rating: uniform ``LINE_MAX_I_KA`` (the case file has none).
* Operating voltage band: [V_MIN, V_MAX] pu (case file: 0.90-1.10; we tighten the upper limit to
  1.05 pu, a common limit for distributed generation, so that PV voltage rise binds).
* Solar parks at ``PV_BUSES``, at the electrically remote ends of the four laterals (confirmed by
  the graph analysis as the highest-impedance-distance buses), rated ``PV_RATED_MW`` each.
* Battery at ``BATTERY_BUS`` (selected in Phase 4 by power-flow-evaluated graph candidates).
* Reverse-power (export) limit at the grid supply point: ``EXPORT_LIMIT_MW``.
The receiving-end bus is bus 0 (grid supply point to the upstream utility).
"""
from __future__ import annotations

import copy

import pandapower as pp
import pandapower.networks as pn

RECEIVING_BUS = 0
LINE_MAX_I_KA = 0.25          # ~5.5 MVA at 12.66 kV
V_MIN, V_MAX = 0.90, 1.05
PV_BUSES = [17, 21, 24, 32]
PV_RATED_MW = 1.5             # per site -> 6 MW total vs 3.715 MW feeder peak load
BATTERY_BUS = 5               # overwritten by grid analysis result if desired (see GRID_MODEL.md)
EXPORT_LIMIT_MW = 1.0
TIE_LINES = [32, 33, 34, 35, 36]  # line indices that are normally open in case33bw

_BASE = None


def base_network():
    """Fresh copy of the 33-bus feeder with assumptions applied and PV/battery elements added (0 MW)."""
    global _BASE
    if _BASE is None:
        net = pn.case33bw()
        net.line["max_i_ka"] = LINE_MAX_I_KA
        net.bus["min_vm_pu"], net.bus["max_vm_pu"] = V_MIN, V_MAX
        net.load["p_base_mw"] = net.load["p_mw"]
        net.load["q_base_mvar"] = net.load["q_mvar"]
        for b in PV_BUSES:
            pp.create_sgen(net, b, p_mw=0.0, q_mvar=0.0, name=f"pv_{b}", type="PV")
        pp.create_sgen(net, BATTERY_BUS, p_mw=0.0, q_mvar=0.0, name="battery", type="storage")
        _BASE = net
    return copy.deepcopy(_BASE)


def open_lines(net) -> list[int]:
    return sorted(net.line.index[~net.line.in_service].tolist())


def set_configuration(net, open_line_ids: list[int]) -> None:
    """Close all lines, then open the given ones (a radial configuration needs exactly 5 open)."""
    net.line["in_service"] = True
    net.line.loc[list(open_line_ids), "in_service"] = False


def set_battery_bus(net, bus: int) -> None:
    idx = net.sgen.index[net.sgen.name == "battery"][0]
    net.sgen.at[idx, "bus"] = bus
