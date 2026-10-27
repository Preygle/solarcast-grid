"""Phase 3: train and compare all forecasting models for all tasks.

Usage: python scripts/train_models.py [--tasks solar_mw:da demand_mw:h1 ...] [--no-lstm]
Outputs: models/<task>/*.joblib, models/registry.json, results/model_comparison.csv,
         results/metrics.json, results/predictions/<task>.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from src.data.preprocess import load_hourly  # noqa: E402
from src.training import train as tr  # noqa: E402

ALL_TASKS = [(t, h) for t in ["solar_mw", "demand_mw", "wind_mw", "net_load_mw"] for h in ["da", "h1"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--no-lstm", action="store_true")
    args = ap.parse_args()
    tasks = [tuple(s.split(":")) for s in args.tasks] if args.tasks else ALL_TASKS
    h = load_hourly()
    comp_path = ROOT / "results" / "model_comparison.csv"
    reg_path = ROOT / "models" / "registry.json"
    comps = [pd.read_csv(comp_path)] if comp_path.exists() and args.tasks else []
    registry = json.loads(reg_path.read_text()) if reg_path.exists() and args.tasks else {}
    for target, horizon in tasks:
        t0 = time.time()
        print(f"== {target} {horizon}", flush=True)
        # Hourly LSTM gets one configuration to bound CPU time; day-ahead gets two.
        tr.LSTM_CONFIGS = [{"hidden": 32, "max_epochs": 25, "patience": 4}, {"hidden": 64, "max_epochs": 25, "patience": 4}] \
            if horizon == "da" else [{"hidden": 64, "max_epochs": 25, "patience": 4}]
        r = tr.run_task(h, target, horizon, run_lstm=not args.no_lstm, log=lambda s: print(s, flush=True))
        task = r["summary"]["task"]
        comps = [c[c.task != task] for c in comps] + [r["comparison"]]
        s = r["summary"]
        registry[task] = {"target": target, "horizon": horizon, "selected_model": s["selected_model"],
                          "artifact": f"models/{task}/{s['selected_model']}.joblib"
                          if s["selected_model"] in tr.FAMILIES + ["lstm"] else None,
                          "pi_level": s["pi_level"], "pi_halfwidth_by_hour": s["pi_halfwidth_by_hour"],
                          "pi_empirical_coverage_test": s["pi_empirical_coverage_test"]}
        out = pd.concat(comps, ignore_index=True)
        out.to_csv(comp_path, index=False)
        reg_path.write_text(json.dumps(registry, indent=2))
        print(f"   selected {s['selected_model']} ({time.time() - t0:.0f}s)", flush=True)

    out = pd.concat(comps, ignore_index=True)
    metrics = {}
    for task, g in out.groupby("task"):
        sel = registry[task]["selected_model"]
        base = g[(g.split == "test") & g.model.isin(["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "samehour_mean_7d"])]
        best_base = base.sort_values("mae").iloc[0]
        selrow = g[(g.split == "test") & (g.model == sel)].iloc[0]
        metrics[task] = {
            "selected_model": sel,
            "selected_test": {k: selrow[k] for k in ["mae", "rmse", "r2", "nmae_pct", "mase"] if k in selrow and pd.notna(selrow[k])},
            "selected_val_mae": float(g[(g.split == "val") & (g.model == sel)].iloc[0]["mae"]),
            "best_baseline": best_base["model"], "best_baseline_test_mae": float(best_base["mae"]),
            "test_mae_improvement_vs_best_baseline_pct": float(100 * (1 - selrow["mae"] / best_base["mae"])),
            "best_test_model_post_hoc": g[g.split == "test"].sort_values("mae").iloc[0]["model"],
            "pi_empirical_coverage_test": registry[task]["pi_empirical_coverage_test"],
        }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=1, default=float))


if __name__ == "__main__":
    main()
