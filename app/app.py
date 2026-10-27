"""SolarCast-Grid research dashboard.  Run:  streamlit run app/app.py

Every number on these pages is read from saved project artefacts or computed live by the project's
own validated code (LP + AC power flow). Provenance tags: OBSERVED, DERIVED, FORECAST, SIMULATED,
OPTIMISED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import data_access as da  # noqa: E402

st.set_page_config(page_title="SolarCast-Grid", layout="wide")
GW = 1e3
TAG = {"OBSERVED": "#2b8a3e", "DERIVED": "#5c7cfa", "FORECAST": "#1971c2", "SIMULATED": "#e8590c", "OPTIMISED": "#9c36b5"}


def tag(*kinds: str) -> None:
    st.markdown(" ".join(f"<span style='background:{TAG[k]};color:white;padding:2px 8px;border-radius:10px;"
                         f"font-size:0.75rem;margin-right:4px'>{k}</span>" for k in kinds), unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("SolarCast-Grid")
    st.caption("Predictive analysis and graph-based optimisation for solar utilisation and power delivery")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
**Research question.** Can day-ahead forecasts of all-India solar and demand, learnt from real Grid-India SCADA data,
inform a constrained battery / flexible-load schedule that measurably reduces solar curtailment and technical losses
on a physically simulated feeder?

**Workflow.** Grid-India SCADA → cleaning & hourly resampling → EDA & duck curve → forecasting (baselines, Ridge,
RF, HistGB, LightGBM, XGBoost, LSTM) → surplus-risk proxy → 33-bus feeder graph & AC power flow →
loss / receiving-end prediction → forecast-informed LP → AC validation → this dashboard.

**Data provenance.**
* **OBSERVED:** Mendeley *y58jknpgs8* v2 (CC BY 4.0), Grid-India/NERLDC all-India SCADA, Sep 2021 – Jun 2025.
  5-min / 10-s data to 2023, hourly report for 2024–25.
* **SIMULATED:** IEEE/Baran-Wu 33-bus feeder (pandapower `case33bw`). It is **not** India's grid; Indian line-level
  topology and receiving-end metering are not publicly available.
* Weather (Open-Meteo archived NWP) is used only in a labelled experiment.
""")
    with c2:
        def status():
            reg = da.js("models/registry.json")
            rows = [{"task": k, "selected model": v["selected_model"],
                     "80% PI coverage (test)": f"{100 * v['pi_empirical_coverage_test']:.0f}%"} for k, v in reg.items()]
            st.markdown("**Model status** (selected by validation MAE)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            h = da.hourly()
            st.markdown(f"**Data:** {len(h):,} hourly rows, {h.index.min():%Y-%m-%d} → {h.index.max():%Y-%m-%d}")
        da.guard(status)
    st.subheader("Limitations to keep in mind")
    st.markdown("""
* All-India aggregates cannot show regional transmission congestion (a major real cause of curtailment in India).
* The surplus-risk metric is an **assumption-based proxy**, not observed curtailment.
* The feeder maps the all-India *shapes* onto a 3.7 MW benchmark feeder; its MW values are not Indian measurements.
* The test period (Jan–Jun 2025) has solar peaks above anything in training. Models use a level-ratio target to cope.
""")


def page_forecasting():
    st.title("Solar & demand forecasting")
    reg = da.js("models/registry.json")
    c = st.columns(3)
    target = c[0].selectbox("Target", ["solar", "demand", "wind", "net_load"])
    horizon = c[1].selectbox("Horizon", ["da", "h1"], format_func=lambda x: {"da": "day-ahead (issued 10:00 D-1)", "h1": "1 hour ahead"}[x])
    task = f"{target}_{horizon}"
    p = da.predictions(task)
    comp = da.csv("results/model_comparison.csv")
    models = [m for m in p.columns if m not in ("y", "split", "pi_lo", "pi_hi")]
    sel = reg[task]["selected_model"]
    chosen = c[2].multiselect("Models", models, default=[sel] + (["seasonal_naive_24h"] if sel != "seasonal_naive_24h" else []))
    t = p[p.split == "test"]
    rng = st.slider("Test window", min_value=t.index.min().to_pydatetime(), max_value=t.index.max().to_pydatetime(),
                    value=(pd.Timestamp("2025-04-07").to_pydatetime(), pd.Timestamp("2025-04-14").to_pydatetime()))
    w = t.loc[rng[0]:rng[1]]
    tag("OBSERVED", "FORECAST")
    fig = go.Figure()
    if sel in chosen:
        fig.add_trace(go.Scatter(x=w.index, y=w.pi_hi / GW, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=w.index, y=w.pi_lo / GW, fill="tonexty", line=dict(width=0), name="80% PI (selected)",
                                 fillcolor="rgba(25,113,194,0.15)"))
    fig.add_trace(go.Scatter(x=w.index, y=w.y / GW, name="actual", line=dict(color="black", width=2)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=w.index, y=w[m] / GW, name=m))
    fig.update_layout(yaxis_title="GW", height=420, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Selected model:** `{sel}` (lowest validation MAE). The prediction interval is split-conformal by hour; "
                f"its empirical test coverage is **{100 * reg[task]['pi_empirical_coverage_test']:.1f}%** against a nominal 80%.")
    g = comp[comp.task == task].pivot_table(index="model", columns="split", values=["mae", "rmse", "r2", "nmae_pct"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    st.dataframe(g.sort_values("mae_val").style.format("{:,.3f}"), use_container_width=True)
    st.caption("MAE/RMSE in MW. Train 2021-09→2024-06, validation 2024-07→12, test 2025-01→06-24 (used once).")

    st.subheader("Live day-ahead forecast from saved models")
    d = st.date_input("Target day", value=pd.Timestamp("2025-05-20"), min_value=pd.Timestamp("2025-01-01"),
                      max_value=pd.Timestamp("2025-06-24"))

    def live():
        fc = da.forecaster().day_ahead(da.hourly(), pd.Timestamp(d))
        act = da.hourly().loc[fc.index]
        f2 = go.Figure()
        for col, color in [("solar_mw", "#e69f00"), ("demand_mw", "#1f4e79"), ("net_load_mw", "#7a1f5c")]:
            f2.add_trace(go.Scatter(x=fc.index, y=fc[col] / GW, name=f"{col} forecast", line=dict(color=color, dash="dash")))
            f2.add_trace(go.Scatter(x=act.index, y=act[col] / GW, name=f"{col} actual", line=dict(color=color)))
        f2.update_layout(yaxis_title="GW", height=380, margin=dict(t=20))
        st.plotly_chart(f2, use_container_width=True)
        st.caption(f"Issued at {fc.issue_time.iloc[0] + pd.Timedelta(hours=1):%Y-%m-%d %H:%M} IST using data up to "
                   f"{fc.issue_time.iloc[0]:%Y-%m-%d %H:%M}. Models: {fc.models.iloc[0]}.")
    da.guard(live)


def page_duck():
    st.title("Duck-curve analysis")
    tag("OBSERVED", "DERIVED")
    h = da.hourly()
    met = da.csv("results/eda/daily_duck_metrics.csv", index_col=0, parse_dates=True)
    c = st.columns(2)
    season = c[0].selectbox("Season", ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"], index=1)
    from src.analysis.duck_curve import seasonal_profiles
    prof = seasonal_profiles(h).xs(season, level="season") / GW
    fig = go.Figure()
    for yr, row in prof.iterrows():
        fig.add_trace(go.Scatter(x=row.index, y=row.values, name=str(yr)))
    fig.update_layout(title=f"Mean net load by hour, {season}", xaxis_title="hour (IST)", yaxis_title="GW", height=380)
    c[0].plotly_chart(fig, use_container_width=True)
    yr = pd.read_csv(da.ROOT / "results/eda/duck_yearly_summary.csv", index_col=0)
    c[1].markdown("**Yearly summary (MW)**")
    c[1].dataframe(yr[["net_load_min_mw", "net_load_max_mw", "mean_evening_ramp_3h_mw", "p95_evening_ramp_3h_mw",
                       "max_evening_ramp_3h_mw", "mean_solar_share_pct", "modal_min_hour"]].round(0), use_container_width=True)
    c[1].caption("Evening ramp = largest 3-hour net-load increase starting 15:00-18:00. Net load = demand - solar - wind.")
    r = met["evening_ramp_3h_mw"] / GW
    f2 = go.Figure([go.Scatter(x=r.index, y=r, mode="lines", line=dict(color="#bbb", width=1), name="daily"),
                    go.Scatter(x=r.index, y=r.rolling(30, center=True).mean(), name="30-day mean", line=dict(color="#7a1f5c", width=3))])
    f2.update_layout(title="Evening 3-hour net-load ramp", yaxis_title="GW / 3 h", height=330)
    st.plotly_chart(f2, use_container_width=True)

    st.subheader("Actual vs forecast net load (day-ahead, test period)")
    tag("OBSERVED", "FORECAST")

    def nl():
        p = da.predictions("net_load_da")
        sel = da.js("models/registry.json")["net_load_da"]["selected_model"]
        d = st.date_input("Day", value=pd.Timestamp("2025-04-12"), min_value=pd.Timestamp("2025-01-01"),
                          max_value=pd.Timestamp("2025-06-24"), key="nl_day")
        w = p.loc[str(d)]
        f3 = go.Figure([go.Scatter(x=w.index.hour, y=w.y / GW, name="actual", line=dict(color="black")),
                        go.Scatter(x=w.index.hour, y=w[sel] / GW, name=f"forecast ({sel})", line=dict(dash="dash"))])
        f3.update_layout(xaxis_title="hour", yaxis_title="GW", height=320)
        st.plotly_chart(f3, use_container_width=True)
    da.guard(nl)

    st.subheader("Surplus-solar risk (proxy)")
    st.markdown(r"""
$S_t(k)=\max\{0,\;k\cdot Solar_t + Wind_t + \theta\cdot Thermal^{committed}_d + Nuclear_t + Hydro^{min}_d - Demand_t\}$

$\theta$ = coal technical minimum (CEA 2023: 55% interim, 40% target); $Thermal^{committed}_d$ = the day's max thermal output;
$k$ = solar scale-up. Uses 2021-09 → 2023-12 only (the generation mix is not reported after that). **This is a scenario
indicator, not observed curtailment.**""")
    sp = da.csv("results/eda/surplus_proxy_scenarios.csv")
    f4 = px.line(sp, x="solar_scale_k", y="surplus_share_of_solar_pct", color=sp.theta.astype(str), markers=True,
                 labels={"color": "theta", "surplus_share_of_solar_pct": "% of scaled solar energy in surplus"})
    f4.update_layout(height=320)
    st.plotly_chart(f4, use_container_width=True)
    st.dataframe(sp[["theta", "solar_scale_k", "surplus_hours", "days_with_surplus", "surplus_energy_gwh",
                     "surplus_share_of_solar_pct", "max_surplus_mw"]].round(2), hide_index=True)


def page_optimisation():
    st.title("Surplus-energy optimisation")
    tag("FORECAST", "SIMULATED", "OPTIMISED")
    from src.grid import network as gn
    from src.optimization.evaluate import run_day
    from src.optimization.lp import BatteryParams, DayInputs, Weights
    from src.simulation.scenarios import map_day, reference_levels

    def aggregate():
        comp = da.csv("results/optimization_comparison.csv")
        piv = comp.pivot(index="metric", columns="case", values="value")
        st.markdown(f"**Saved evaluation over {int(comp.days.iloc[0])} test days (all four cases solved optimally on the "
                    f"same days).** Losses, energies and voltages come from AC power flow.")
        keep = ["pv_available_mwh", "curtailment_mwh", "solar_utilisation_pct", "loss_mwh", "received_export_mwh",
                "import_mwh", "mean_daily_peak_import_mw", "mean_evening_ramp_3h_mw", "battery_throughput_mwh",
                "v_violation_hours", "thermal_violation_hours", "export_violation_hours", "lp_slack_total"]
        st.dataframe(piv.reindex(keep)[["baseline", "rule_based", "forecast_informed", "perfect_foresight"]].round(3),
                     use_container_width=True)
    da.guard(aggregate)

    st.subheader("Run the optimiser for a day")
    dec = da.js("results/grid_scenarios/grid_decisions.json")
    c = st.columns(5)
    day = c[0].date_input("Test day", value=pd.Timestamp("2025-04-12"), min_value=pd.Timestamp("2025-01-01"),
                          max_value=pd.Timestamp("2025-06-24"), key="opt_day")
    e = c[1].slider("Battery energy [MWh]", 0.0, 16.0, 6.0, 0.5)
    pw = c[2].slider("Battery power [MW]", 0.0, 4.0, 1.5, 0.25)
    soc0 = c[3].slider("Initial SoC", 0.1, 0.9, 0.5, 0.05)
    flex = c[4].slider("Flexible load ±share", 0.0, 0.3, 0.10, 0.01)
    c2 = st.columns(3)
    pv_scale = c2[0].slider("PV capacity multiplier (×1.5 MW per site)", 0.2, 2.0, 1.0, 0.1)
    exp_lim = c2[1].slider("Export limit at receiving bus [MW]", 0.0, 4.0, gn.EXPORT_LIMIT_MW, 0.25)
    use_fc = c2[2].radio("Plan with", ["day-ahead forecast", "perfect foresight"], horizontal=True)
    if st.button("Optimise and validate with AC power flow", type="primary"):
        def go_run():
            h = da.hourly()
            refs = reference_levels(h)
            idx = pd.date_range(pd.Timestamp(day), periods=24, freq="h")
            act = map_day(h.loc[idx, "demand_mw"], h.loc[idx, "solar_mw"], refs)
            fc = da.forecaster().day_ahead(h, pd.Timestamp(day))
            fct = map_day(fc["demand_mw"], fc["solar_mw"], refs)
            mk = lambda prof: DayInputs(prof.load_scale.to_numpy(), np.outer(prof.pv_pu, np.full(4, gn.PV_RATED_MW * pv_scale)),
                                        flex_frac=flex, export_limit_mw=exp_lim)
            net = gn.base_network()
            gn.set_battery_bus(net, dec["battery_bus"])
            bp = BatteryParams(bus=dec["battery_bus"], power_mw=pw, energy_mwh=e, soc_init=soc0)
            case = "forecast_informed" if use_fc.startswith("day") else "perfect_foresight"
            with st.spinner("Solving LP and running 48 AC power flows..."):
                res = run_day(net, mk(act), mk(fct), bp, Weights(), cases=["baseline", case])
            if any(r.get("realise_status") != "Optimal" for r in res.values()):
                st.error(f"Solver status: { {k: (v.get('plan_status'), v.get('realise_status')) for k, v in res.items()} }")
                return
            b, o = res["baseline"], res[case]
            m = pd.DataFrame({"baseline": b, case: o}).loc[["pv_available_mwh", "curtailment_mwh", "pv_used_mwh", "loss_mwh",
                                                            "received_export_mwh", "import_mwh", "peak_import_mw",
                                                            "evening_ramp_3h_mw", "vm_max_pu", "vm_min_pu",
                                                            "v_violation_hours", "thermal_violation_hours",
                                                            "export_violation_hours", "lp_slack_total"]].astype(float)
            m["change"] = m[case] - m["baseline"]
            st.dataframe(m.round(4), use_container_width=True)
            st.caption(f"Plan status: {o['plan_status']}; realisation: {o['realise_status']}. Slack > 0 means the "
                       f"forecast-based schedule needed constraint relaxation on the actual day; this counts as a violation.")
            s = o["schedule"]
            hrs = list(range(24))
            f = go.Figure()
            f.add_trace(go.Bar(x=hrs, y=s["ch_mw"], name="charge MW"))
            f.add_trace(go.Bar(x=hrs, y=-s["dis_mw"], name="discharge MW (neg)"))
            f.add_trace(go.Scatter(x=hrs, y=s["soc_mwh"], name="SoC MWh", yaxis="y2"))
            f.add_trace(go.Scatter(x=hrs, y=act.pv_pu * 4 * gn.PV_RATED_MW * pv_scale, name="PV available (actual)", line=dict(color="#e69f00")))
            f.add_trace(go.Scatter(x=hrs, y=fct.pv_pu * 4 * gn.PV_RATED_MW * pv_scale, name="PV available (forecast)", line=dict(color="#e69f00", dash="dash")))
            f.add_trace(go.Scatter(x=hrs, y=-b["hourly"]["p_recv_mw"], name="baseline import at bus 0 (AC)", line=dict(color="#888")))
            f.add_trace(go.Scatter(x=hrs, y=-o["hourly"]["p_recv_mw"], name=f"{case} import at bus 0 (AC)", line=dict(color="#9c36b5")))
            f.update_layout(barmode="relative", height=420, xaxis_title="hour", yaxis_title="MW",
                            yaxis2=dict(overlaying="y", side="right", title="MWh"))
            st.plotly_chart(f, use_container_width=True)
        da.guard(go_run)
    with st.expander("Formulation"):
        st.markdown((da.ROOT / "docs" / "OPTIMIZATION_METHODOLOGY.md").read_text(encoding="utf-8")
                    if (da.ROOT / "docs" / "OPTIMIZATION_METHODOLOGY.md").exists() else "See src/optimization/lp.py")


def page_grid():
    st.title("Grid graph & power delivery")
    tag("SIMULATED")
    from src.grid import graph as G
    from src.grid import network as gn
    from src.power_flow.runner import OperatingPoint, run
    from src.receiving_power.model import feature_frame
    from src.simulation.scenarios import ZONES

    dec = da.js("results/grid_scenarios/grid_decisions.json")
    summ = da.js("results/grid_scenarios/grid_summary.json")
    names = dec["config_names"]
    cfgs = dec["configs"]
    labels = [f"{i}: {names[i] if i < len(names) else 'config'} (open {'-'.join(map(str, c))})" for i, c in enumerate(cfgs)]
    c = st.columns(4)
    ci = c[0].selectbox("Configuration", range(len(cfgs)), format_func=lambda i: labels[i])
    load = c[1].slider("Load scale (× base 3.715 MW)", 0.3, 1.3, 0.8, 0.05)
    pv = c[2].slider("PV output per site [MW]", 0.0, 2.25, 1.0, 0.05)
    batt = c[3].slider("Battery [MW] (+discharge)", -2.0, 2.0, 0.0, 0.1)
    net = gn.base_network()
    gn.set_battery_bus(net, dec["battery_bus"])
    gn.set_configuration(net, cfgs[ci])
    res = run(net, OperatingPoint(load, np.full(4, pv), battery_mw=batt))

    full = G.to_graph(net)
    pos = {b: tuple(json.loads(net.bus.at[b, "geo"])["coordinates"]) for b in net.bus.index}
    ex, ey, ox, oy = [], [], [], []
    for u, v, d in full.edges(data=True):
        xs, ys = (ex, ey) if d["line"] not in cfgs[ci] else (ox, oy)
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    vm = net.res_bus.vm_pu if res["converged"] else pd.Series(1.0, index=net.bus.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#444", width=2), name="closed line"))
    fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="#bbb", width=1, dash="dash"), name="open line"))
    fig.add_trace(go.Scatter(x=[pos[b][0] for b in net.bus.index], y=[pos[b][1] for b in net.bus.index], mode="markers+text",
                             text=[str(b) for b in net.bus.index], textposition="top center",
                             marker=dict(size=14, color=vm, colorscale="RdYlBu_r", cmin=0.9, cmax=1.06,
                                         colorbar=dict(title="V [pu]"), line=dict(width=[3 if b in gn.PV_BUSES + [dec['battery_bus'], 0] else 0 for b in net.bus.index], color="#e69f00")),
                             name="bus (colour = AC voltage)"))
    fig.update_layout(height=480, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Receiving bus 0 (grid supply point). PV at buses {gn.PV_BUSES}; battery at bus {dec['battery_bus']}. "
               f"{summ['radial_configurations_matrix_tree']:,} radial configurations exist (matrix-tree theorem).")

    if not res["converged"]:
        st.error("AC power flow did not converge for this operating point. No result is shown.")
        return
    # model prediction for the same operating point
    try:
        mods = da.receiving_models()
        row = {"config_id": ci, "battery_mw": batt, "load_p_total_mw": net.load.p_mw.sum(),
               "load_q_total_mvar": net.load.q_mvar.sum(), **{f"pv_{b}_mw": pv for b in gn.PV_BUSES}}
        for z, buses in ZONES.items():
            row[f"load_{z}_mw"] = float(net.load.loc[net.load.bus.isin(buses), "p_mw"].sum())
        df = pd.DataFrame([row])
        anym = next(iter(mods.values()))
        X = feature_frame(df, len(anym["configs"]), anym["config_features"])[anym["features"]]
        pred = {t: float(m["model"].predict(X)[0]) for t, m in mods.items()}
    except da.ArtefactMissing as e:
        st.warning(str(e))
        pred = {}
    cols = st.columns(4)
    cols[0].metric("Received at bus 0 (AC PF, SIMULATED)", f"{res['p_recv_mw']:.3f} MW",
                   delta=f"model: {pred.get('p_recv_mw', float('nan')):.3f} MW", delta_color="off")
    cols[1].metric("Technical losses (AC PF)", f"{1e3 * res['p_loss_mw']:.1f} kW",
                   delta=f"model: {1e3 * pred.get('p_loss_mw', float('nan')):.1f} kW", delta_color="off")
    cols[2].metric("Loss % of sent power", f"{res['loss_pct']:.2f} %",
                   delta=f"model: {pred.get('loss_pct', float('nan')):.2f} %", delta_color="off")
    cols[3].metric("Voltage range", f"{res['vm_min_pu']:.3f} – {res['vm_max_pu']:.3f} pu",
                   delta="VIOLATION" if res["v_violation"] else "within 0.90–1.05", delta_color="inverse" if res["v_violation"] else "off")
    st.caption(f"Sent {res['p_sent_mw']:.3f} MW = delivered {res['p_delivered_mw']:.3f} MW + losses {res['p_loss_mw']:.4f} MW. "
               f"Energy-balance residual {res['balance_residual_mw']:.1e} MW. Max line loading {res['line_loading_max_pct']:.1f}% "
               f"(assumed rating {gn.LINE_MAX_I_KA} kA).")
    st.subheader("Reconfiguration and battery siting (graph candidates evaluated by AC power flow)")
    st.dataframe(da.csv("results/grid_scenarios/reconfiguration.csv").round(4), hide_index=True, use_container_width=True)
    st.json({k: summ[k] for k in ["battery_bus_selected", "graph_top5_by_loss_moment", "spearman_graph_metric_vs_pf_loss_reduction"]})


def page_models():
    st.title("Model comparison & evaluation")
    tag("FORECAST", "SIMULATED")
    m = da.js("results/metrics.json")
    rows = [{"task": k, "selected": v["selected_model"], "test MAE (MW)": v["selected_test"]["mae"],
             "test R²": v["selected_test"]["r2"], "nMAE %": v["selected_test"]["nmae_pct"],
             "best baseline": v["best_baseline"], "baseline MAE (MW)": v["best_baseline_test_mae"],
             "improvement %": v["test_mae_improvement_vs_best_baseline_pct"],
             "best on test (post hoc)": v["best_test_model_post_hoc"],
             "PI coverage": v["pi_empirical_coverage_test"]} for k, v in m.items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption("Selection uses validation (Jul–Dec 2024) only. 'Best on test' is shown for transparency and was not used.")
    for img, cap in [("results/figures/fc_model_comparison.png", "Validation vs test MAE for every model"),
                     ("results/figures/fc_mae_by_hour.png", "Test MAE by hour of day (selected vs best baseline)"),
                     ("results/figures/fc_residuals_dayahead.png", "Residual distributions (day-ahead, test)"),
                     ("results/figures/fc_permutation_importance.png", "Permutation importance on the test set")]:
        try:
            st.image(str(da.image(img)), caption=cap)
        except da.ArtefactMissing as e:
            st.warning(str(e))

    def abl():
        a = da.csv("results/feature_ablation_weather.csv")
        st.subheader("Feature ablation and weather experiment (LightGBM, day-ahead)")
        st.dataframe(a.pivot_table(index=["target", "variant"], columns="split", values="mae").round(0), use_container_width=True)
        st.caption("Weather = Open-Meteo archived short-lead NWP. This is optimistic for a real 10:00 D-1 issue time.")
    da.guard(abl)

    st.subheader("Loss and receiving-end power models (SIMULATED labels, held-out scenario families)")
    def grid_models():
        g = da.csv("results/grid_model_comparison.csv")
        st.dataframe(g.pivot_table(index="model", columns=["target", "split"], values="mae").round(5), use_container_width=True)
        st.json(da.js("results/grid_scenarios/fresh_pf_check.json"))
        st.image(str(da.image("results/figures/grid_receiving_power_pred_vs_sim.png")))
    da.guard(grid_models)
    st.markdown("""
**Assumptions & limitations.** Chronological splits only; the test set was evaluated once after selection.
Tree ensembles were trained on level-normalised targets because test-period solar exceeds the training range.
Grid-model labels are SIMULATED by AC power flow on a benchmark feeder, so their accuracy says nothing about real
Indian receiving-end metering.""")


PAGES = {"Overview": page_overview, "Forecasting": page_forecasting, "Duck curve": page_duck,
         "Optimisation": page_optimisation, "Grid & power delivery": page_grid, "Model evaluation": page_models}
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption("Provenance tags: " + ", ".join(TAG))
try:
    PAGES[choice]()
except da.ArtefactMissing as e:
    st.warning(str(e))
