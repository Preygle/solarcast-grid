"""Phase 4 (Problem B): simulate labelled scenarios, train loss / receiving-end models, evaluate on
held-out scenario families, and check predictions against FRESH power-flow runs.

Usage: python scripts/run_receiving_power.py [--families 500]
Outputs: results/grid_scenarios/scenarios.parquet, results/grid_model_comparison.csv,
         results/grid_scenarios/fresh_pf_check.csv, models/receiving_power/*.joblib, figures.
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

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import load_hourly  # noqa: E402
from src.grid.graph import node_metrics  # noqa: E402
from src.grid import network as gn  # noqa: E402
from src.receiving_power import model as rp  # noqa: E402
from src.simulation.scenarios import generate  # noqa: E402

OUT = ROOT / "results" / "grid_scenarios"
FIG = ROOT / "results" / "figures"
MODEL_DIR = ROOT / "models" / "receiving_power"


def config_features(configs) -> pd.DataFrame:
    """Graph features of each configuration: resistance distance from each PV bus to bus 0."""
    rows = {}
    for k, c in enumerate(configs):
        net = gn.base_network()
        gn.set_configuration(net, c)
        nm = node_metrics(net)
        rows[k] = {f"rdist_pv_{b}": nm.at[b, "r_dist_ohm"] for b in gn.PV_BUSES}
        rows[k]["rdist_mean_all"] = nm["r_dist_ohm"].mean()
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=500)
    args = ap.parse_args()
    dec = json.loads((OUT / "grid_decisions.json").read_text())
    configs, batt_bus = dec["configs"], dec["battery_bus"]
    h = load_hourly()

    t0 = time.time()
    df = generate(h, configs, args.families, seed=0, battery_bus=batt_bus)
    n_total = len(df)
    conv = df["converged"].astype(bool)
    print(f"simulated {n_total} operating points in {time.time() - t0:.0f}s; converged {conv.sum()}")
    df.to_parquet(OUT / "scenarios.parquet")
    df = df[conv].reset_index(drop=True)
    print("balance residual max |MW|:", float(df.balance_residual_mw.abs().max()))
    print(df[["p_loss_mw", "p_recv_mw", "loss_pct", "load_p_total_mw"]].describe().round(4))

    cf = config_features(configs)
    X = rp.feature_frame(df, len(configs), cf)
    tr, va, te = rp.group_split(df)
    assert not set(df.loc[tr, "family"]) & set(df.loc[te, "family"])
    rows, preds, fitted = [], {}, {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for tgt in rp.TARGETS:
        y = df[tgt]
        ok = y.notna()
        for name, mdl in rp.models().items():
            m = mdl.fit(X.loc[tr][ok[tr]], y.loc[tr][ok[tr]])
            for part, idx in [("val", va), ("test", te)]:
                ii = idx[ok[idx]]
                p = m.predict(X.loc[ii])
                rows.append({"target": tgt, "model": name, "split": part, "n": len(ii),
                             "label_source": "SIMULATED", **rp.metrics(y.loc[ii], p)})
                if part == "test":
                    preds[(tgt, name)] = pd.Series(p, index=ii)
            fitted[(tgt, name)] = m
    comp = pd.DataFrame(rows)
    comp.to_csv(ROOT / "results" / "grid_model_comparison.csv", index=False)
    best = {}
    for tgt in rp.TARGETS:
        v = comp[(comp.target == tgt) & (comp.split == "val")].sort_values("mae").iloc[0]
        best[tgt] = v["model"]
        joblib.dump({"model": fitted[(tgt, v["model"])], "features": list(X.columns), "configs": configs,
                     "config_features": cf, "battery_bus": batt_bus, "target": tgt},
                    MODEL_DIR / f"{tgt}.joblib")
    print(comp.round(4).to_string())
    print("selected (val MAE):", best)

    # Fresh power-flow check: new scenario families with a different seed (never simulated before).
    fresh = generate(h, configs, 40, seed=12345, battery_bus=batt_bus)
    fresh = fresh[fresh.converged.astype(bool)].reset_index(drop=True)
    Xf = rp.feature_frame(fresh, len(configs), cf)
    chk = {"n_points": len(fresh)}
    fresh_out = fresh[["family", "hour", "config_id"]].copy()
    for tgt in rp.TARGETS:
        m = fitted[(tgt, best[tgt])]
        ok = fresh[tgt].notna()
        p = m.predict(Xf[ok])
        fresh_out.loc[ok, f"{tgt}_simulated"] = fresh.loc[ok, tgt]
        fresh_out.loc[ok, f"{tgt}_predicted"] = p
        chk[tgt] = {"model": best[tgt], **rp.metrics(fresh.loc[ok, tgt], p)}
    fresh_out.to_csv(OUT / "fresh_pf_check.csv", index=False)
    (OUT / "fresh_pf_check.json").write_text(json.dumps(chk, indent=2))
    print("fresh PF check:", json.dumps(chk, indent=1))

    # Figures: predicted vs simulated on held-out families
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, tgt in zip(axes, rp.TARGETS):
        p = preds[(tgt, best[tgt])]
        y = df.loc[p.index, tgt]
        ax.scatter(y, p, s=4, alpha=0.4)
        lim = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1)
        mt = comp[(comp.target == tgt) & (comp.model == best[tgt]) & (comp.split == "test")].iloc[0]
        ax.set(xlabel=f"SIMULATED {rp.TARGETS[tgt]}", ylabel="model PREDICTED",
               title=f"{best[tgt]}: MAE {mt.mae:.4f}, R2 {mt.r2:.4f} (held-out families)")
    fig.tight_layout()
    fig.savefig(FIG / "grid_receiving_power_pred_vs_sim.png", dpi=120)
    plt.close(fig)

    lt = comp[(comp.split == "test")].pivot(index="model", columns="target", values="mae")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    lt.plot.bar(ax=ax, logy=True)
    ax.set(ylabel="test MAE (unit of target, log)", title="Loss / receiving-end model comparison (held-out families, SIMULATED labels)")
    fig.tight_layout()
    fig.savefig(FIG / "grid_model_comparison.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
