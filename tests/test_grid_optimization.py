import numpy as np
import pytest

from src.grid import graph as G
from src.grid import network as gn
from src.optimization.evaluate import ac_validate, rule_based_schedule
from src.optimization.lp import BatteryParams, DayInputs, Fixed, solve_day
from src.power_flow.runner import OperatingPoint, run


def _day(pv_peak=1.4, load=0.75):
    hrs = np.arange(24)
    pv_pu = np.clip(np.sin(np.pi * (hrs - 6) / 12), 0, None)
    avail = np.outer(pv_pu, np.full(4, pv_peak))
    return DayInputs(np.full(24, load), avail, flex_frac=0.1)


def test_base_case_matches_published_losses():
    net = gn.base_network()
    r = run(net, OperatingPoint(1.0, np.zeros(4)))
    assert r["converged"]
    assert r["p_loss_mw"] == pytest.approx(0.2027, abs=5e-4)     # Baran & Wu 33-bus base case
    assert r["vm_min_pu"] == pytest.approx(0.913, abs=1e-3)


def test_energy_balance_and_sign_conventions():
    net = gn.base_network()
    r = run(net, OperatingPoint(0.5, np.full(4, 1.5), battery_mw=-0.5))
    # sources - sinks - losses - export == 0, enforced inside run(); check definitions here
    assert r["p_recv_mw"] > 0 and r["p_export_mw"] == pytest.approx(r["p_recv_mw"])
    assert r["p_delivered_mw"] == pytest.approx(r["p_sent_mw"] - r["p_loss_mw"])
    assert r["p_pv_mw"] + r["p_batt_mw"] - r["p_load_mw"] - r["p_loss_mw"] - r["p_recv_mw"] == pytest.approx(0, abs=1e-6)
    assert 0 < r["loss_pct"] < 20


def test_graph_structure():
    net = gn.base_network()
    assert G.is_radial(net, gn.open_lines(net))
    assert not G.is_radial(net, [32, 33, 34, 35])           # one tie closed -> loop
    assert G.count_radial_configs(net) == 50751
    assert len(G.loops(net)) == 5
    nm = G.node_metrics(net)
    assert nm.loc[0, "r_dist_ohm"] == 0 and nm["r_dist_ohm"].idxmax() == 17
    for cfg in G.one_switch_neighbours(net, gn.open_lines(net))[:10]:
        assert G.is_radial(net, cfg)


def test_lp_feasible_and_battery_physics():
    net = gn.base_network()
    gn.set_battery_bus(net, 17)
    b = BatteryParams(bus=17)
    inp = _day()
    base = solve_day(net, DayInputs(inp.load_scale, inp.pv_avail_mw, flex_frac=0.0), None)
    opt = solve_day(net, inp, b)
    assert base.status == opt.status == "Optimal"
    s = opt.schedule
    soc = s["soc_mwh"]
    prev = np.concatenate([[b.soc_init * b.energy_mwh], soc[:-1]])
    assert np.allclose(soc, prev + b.eta * s["ch_mw"] - s["dis_mw"] / b.eta, atol=1e-6)
    assert soc.min() >= b.soc_min * b.energy_mwh - 1e-6 and soc.max() <= b.soc_max * b.energy_mwh + 1e-6
    assert soc[-1] >= b.soc_init * b.energy_mwh - 1e-6
    assert (s["ch_mw"] <= b.power_mw + 1e-6).all() and (s["dis_mw"] <= b.power_mw + 1e-6).all()
    # flexible load is energy-neutral over the day
    assert abs(np.sum(s["flex_mw"])) < 1e-6
    assert (np.abs(s["flex_mw"]) <= inp.flex_frac * inp.load_scale * 3.715 + 1e-6).all()
    # PV used never exceeds availability; storage can only reduce required curtailment
    assert (s["pv_used_mw"] <= inp.pv_avail_mw + 1e-6).all()
    assert opt.terms["curtailment_mwh"] <= base.terms["curtailment_mwh"] + 1e-6
    # export limit respected in the LP
    assert (-s["head_mw"] <= inp.export_limit_mw + 1e-6).all()


def test_ac_validation_respects_limits_for_lp_schedule():
    net = gn.base_network()
    gn.set_battery_bus(net, 17)
    inp = _day()
    opt = solve_day(net, inp, BatteryParams(bus=17))
    ac = ac_validate(net, inp, opt.schedule)
    assert ac["ac_converged_hours"] == 24
    assert ac["v_violation_hours"] == 0 and ac["thermal_violation_hours"] == 0
    # LinDistFlow ignores losses, so AC export can differ slightly; it must stay near the limit
    assert ac["hourly"]["p_export_mw"].max() <= inp.export_limit_mw + 0.05


def test_forecast_plan_realisable_on_different_actual_day():
    """A plan made on one profile must be realisable (status Optimal) on a different actual day;
    any constraint relief shows up as reported slack, never as infeasibility."""
    net = gn.base_network()
    gn.set_battery_bus(net, 17)
    plan = solve_day(net, _day(pv_peak=1.4, load=0.75), BatteryParams(bus=17))
    fx = Fixed(plan.schedule["ch_mw"], plan.schedule["dis_mw"], plan.schedule["flex_mw"])
    real = solve_day(net, _day(pv_peak=1.0, load=0.9), BatteryParams(bus=17), fixed=fx)
    assert real.status == "Optimal"


def test_realisation_with_fixed_decisions_reports_slack():
    net = gn.base_network()
    gn.set_battery_bus(net, 17)
    inp = _day()
    plan = solve_day(net, inp, BatteryParams(bus=17))
    fx = Fixed(plan.schedule["ch_mw"], plan.schedule["dis_mw"], plan.schedule["flex_mw"])
    real = solve_day(net, inp, BatteryParams(bus=17), fixed=fx)
    assert real.status == "Optimal"
    assert real.terms["slack_total"] == pytest.approx(0, abs=1e-6)  # same day -> plan is feasible
    assert real.terms["curtailment_mwh"] == pytest.approx(plan.terms["curtailment_mwh"], abs=1e-4)


def test_rule_based_schedule_bounds():
    b = BatteryParams()
    fx = rule_based_schedule(_day(), b, base_load_mw=3.715)
    assert (fx.ch >= 0).all() and (fx.ch <= b.power_mw + 1e-9).all() and (fx.dis <= b.power_mw + 1e-9).all()
    # charging only from surplus above the export limit; discharge only in the evening window
    assert (fx.dis[:16] == 0).all() and (fx.dis[22:] == 0).all()
