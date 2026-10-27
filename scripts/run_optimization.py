"""Phase 4 (Problem A): forecast-informed optimisation vs baselines on every test day, AC-validated.

Usage: python scripts/run_optimization.py [--days N] [--no-sensitivity]
Outputs: results/optimization_comparison.csv (totals per case + change vs baseline),
         results/grid_scenarios/optimization_daily.csv, optimization_sensitivity.csv, figures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import load_hourly  # noqa: E402
from src.features.splits import SPLITS  # noqa: E402
from src.grid import network as gn  # noqa: E402
from src.inference.forecast import Forecaster  # noqa: E402
from src.optimization.evaluate import CASES, run_day  # noqa: E402
from src.optimization.lp import BatteryParams, DayInputs, Weights  # noqa: E402
from src.simulation.scenarios import map_day, reference_levels  # noqa: E402

OUT = ROOT / "results" / "grid_scenarios"
FIG = ROOT / "results" / "figures"
METRICS = ["pv_available_mwh", "curtailment_mwh", "pv_used_mwh", "loss_mwh", "received_export_mwh",
           "import_mwh", "net_received_mwh", "delivered_mwh", "peak_import_mw", "evening_ramp_3h_mw",
           "battery_throughput_mwh", "v_violation_hours", "thermal_violation_hours", "export_violation_hours",
           "lp_slack_total"]


def day_inputs(prof: pd.DataFrame, flex: float) -> DayInputs:
    avail = np.outer(prof["pv_pu"].to_numpy(), np.full(len(gn.PV_BUSES), gn.PV_RATED_MW))
    return DayInputs(prof["load_scale"].to_numpy(), avail, flex_frac=flex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="limit number of test days (0 = all)")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()
    dec = json.loads((OUT / "grid_decisions.json").read_text())
    batt = BatteryParams(bus=dec["battery_bus"])
    net = gn.base_network()
    gn.set_battery_bus(net, batt.bus)

    h = load_hourly()
    refs = reference_levels(h)
    a, b = SPLITS["test"]
    days = pd.date_range(a, pd.Timestamp(b).normalize(), freq="D")
    days = [d for d in days if h.loc[d: d + pd.Timedelta(hours=23), ["demand_mw", "solar_mw"]].notna().all().all()
            and len(h.loc[d: d + pd.Timedelta(hours=23)]) == 24]
    if args.days:
        days = days[:: max(1, len(days) // args.days)][: args.days]

    fc = Forecaster()
    when = pd.DatetimeIndex([d + pd.Timedelta(hours=k) for d in days for k in range(24)])
    f_sol = fc.predict(h, "solar_mw", "da", when)
    f_dem = fc.predict(h, "demand_mw", "da", when)
    print(f"{len(days)} test days; forecast models: solar={f_sol.model.iloc[0]}, demand={f_dem.model.iloc[0]}", flush=True)
    fc_mae = {"solar_mw": float((f_sol.solar_mw - h.loc[when, "solar_mw"]).abs().mean()),
              "demand_mw": float((f_dem.demand_mw - h.loc[when, "demand_mw"]).abs().mean())}

    rows, hourly_store = [], {}
    t0 = time.time()
    w = Weights()
    for i, d in enumerate(days):
        idx = pd.date_range(d, periods=24, freq="h")
        act = map_day(h.loc[idx, "demand_mw"], h.loc[idx, "solar_mw"], refs)
        fct = map_day(f_dem.loc[idx, "demand_mw"], f_sol.loc[idx, "solar_mw"], refs)
        res = run_day(net, day_inputs(act, 0.10), day_inputs(fct, 0.10), batt, w)
        for case, r in res.items():
            rows.append({"date": d.date().isoformat(), "case": case,
                         "plan_status": r.get("plan_status"), "realise_status": r.get("realise_status"),
                         **{m: r.get(m, np.nan) for m in METRICS}})
            if "hourly" in r and d.date().isoformat() in ("2025-04-12", "2025-01-15", "2025-06-10"):
                hourly_store[(d.date().isoformat(), case)] = (r["hourly"], r["schedule"], act, fct)
        if i % 20 == 19:
            print(f"   {i + 1}/{len(days)} days ({time.time() - t0:.0f}s)", flush=True)
    daily = pd.DataFrame(rows)
    daily.to_csv(OUT / "optimization_daily.csv", index=False)

    ok = daily.groupby("date")["realise_status"].apply(lambda s: (s == "Optimal").all())
    valid_days = ok[ok].index
    dv = daily[daily.date.isin(valid_days)]
    tot = dv.groupby("case")[METRICS].sum()
    tot.loc[:, "peak_import_mw"] = dv.groupby("case")["peak_import_mw"].max()
    tot["mean_daily_peak_import_mw"] = dv.groupby("case")["peak_import_mw"].mean()
    tot["mean_evening_ramp_3h_mw"] = dv.groupby("case")["evening_ramp_3h_mw"].mean()
    tot = tot.drop(columns=["evening_ramp_3h_mw"])
    tot["solar_utilisation_pct"] = 100 * tot["pv_used_mwh"] / tot["pv_available_mwh"]
    tot["days"] = len(valid_days)
    tot = tot.reindex(CASES)
    base = tot.loc["baseline"]
    comp = []
    for case in CASES:
        for m in tot.columns:
            if m == "days":
                continue
            v, b0 = tot.at[case, m], base[m]
            comp.append({"case": case, "metric": m, "value": v, "baseline": b0, "abs_change": v - b0,
                         "pct_change": 100 * (v - b0) / abs(b0) if b0 not in (0, np.nan) and abs(b0) > 1e-9 else np.nan})
    comp = pd.DataFrame(comp)
    comp["days"] = len(valid_days)
    comp["data_class"] = "SIMULATED feeder; OBSERVED all-India shapes; FORECAST-informed decisions"
    comp.to_csv(ROOT / "results" / "optimization_comparison.csv", index=False)
    meta = {"days_total": len(days), "days_all_cases_optimal": int(len(valid_days)), "battery": asdict(batt),
            "weights": asdict(w), "export_limit_mw": gn.EXPORT_LIMIT_MW, "pv_rated_mw_per_site": gn.PV_RATED_MW,
            "flex_frac": 0.10, "forecast_mae_mw_on_these_days": fc_mae,
            "status_counts": daily.groupby(["case", "realise_status"]).size().astype(int).reset_index().values.tolist()}
    (OUT / "optimization_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tot.round(3).T.to_string())
    print(json.dumps(meta, indent=1, default=str))

    plot_days(hourly_store)
    if not args.no_sensitivity:
        sensitivity(h, refs, net, batt, days, f_sol, f_dem)


def plot_days(store):
    dates = sorted({k[0] for k in store})
    if not dates:
        return
    fig, axes = plt.subplots(len(dates), 2, figsize=(14, 3.6 * len(dates)), squeeze=False)
    for row, d in zip(axes, dates):
        for case, col in [("baseline", "#888"), ("forecast_informed", "#1f77b4"), ("perfect_foresight", "#2ca02c")]:
            if (d, case) not in store:
                continue
            hr, sc, act, fct = store[(d, case)]
            row[0].plot(hr["hour"], -hr["p_recv_mw"], color=col, label=f"{case}: feeder net import (AC)")
            if case != "baseline":
                row[1].plot(range(24), sc["soc_mwh"], color=col, label=f"{case} SoC [MWh]")
        hr, sc, act, fct = store[(d, "baseline")]
        row[0].axhline(-gn.EXPORT_LIMIT_MW, color="r", ls=":", label="export limit")
        row[0].set(title=f"{d}: power at receiving bus 0 (negative = export to utility)", xlabel="hour", ylabel="MW")
        row[1].plot(range(24), act["pv_pu"] * gn.PV_RATED_MW * 4, "k-", lw=1, label="PV available (actual)")
        row[1].plot(range(24), fct["pv_pu"] * gn.PV_RATED_MW * 4, "k--", lw=1, label="PV available (forecast)")
        row[1].set(title="battery state of charge and PV availability", xlabel="hour")
        row[0].legend(fontsize=7); row[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "optimization_example_days.png", dpi=110)
    plt.close(fig)


def sensitivity(h, refs, net, batt, days, f_sol, f_dem):
    sub = days[:: max(1, len(days) // 20)][:20]
    grid = [("ramp", v) for v in [0.0, 0.05, 0.2, 1.0]] + [("loss", v) for v in [0.0, 3.0]] + \
           [("battery_mwh", v) for v in [3.0, 12.0]]
    rows = []
    for kind, val in grid:
        w = Weights()
        bp = BatteryParams(bus=batt.bus)
        if kind == "ramp":
            w.ramp = val
        elif kind == "loss":
            w.loss = val
        else:
            bp.energy_mwh, bp.power_mw = val, val / 4
        for d in sub:
            idx = pd.date_range(d, periods=24, freq="h")
            act = map_day(h.loc[idx, "demand_mw"], h.loc[idx, "solar_mw"], refs)
            fct = map_day(f_dem.loc[idx, "demand_mw"], f_sol.loc[idx, "solar_mw"], refs)
            res = run_day(net, day_inputs(act, 0.10), day_inputs(fct, 0.10), bp, w, cases=["forecast_informed"])
            r = res["forecast_informed"]
            rows.append({"param": kind, "value": val, "date": d.date().isoformat(),
                         **{m: r.get(m, np.nan) for m in ["curtailment_mwh", "loss_mwh", "evening_ramp_3h_mw",
                                                          "peak_import_mw", "battery_throughput_mwh", "lp_slack_total"]}})
    s = pd.DataFrame(rows)
    agg = s.groupby(["param", "value"]).agg(curtailment_mwh=("curtailment_mwh", "sum"), loss_mwh=("loss_mwh", "sum"),
                                            mean_evening_ramp_3h_mw=("evening_ramp_3h_mw", "mean"),
                                            mean_peak_import_mw=("peak_import_mw", "mean"),
                                            throughput_mwh=("battery_throughput_mwh", "sum"), days=("date", "count"))
    agg.to_csv(OUT / "optimization_sensitivity.csv")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
