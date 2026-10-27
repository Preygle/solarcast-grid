"""Phase 3 evaluation: figures, horizon / hour / regime breakdowns, feature importance,
weather experiment (E4) and feature ablation. Requires scripts/train_models.py outputs.

Usage: python scripts/evaluate_models.py [--skip-experiments]
"""
from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.inspection import permutation_importance  # noqa: E402

from src.data.preprocess import load_hourly  # noqa: E402
from src.data.weather import load_weather  # noqa: E402
from src.features.build import build_features, default_spec  # noqa: E402
from src.features.splits import split_mask  # noqa: E402
from src.models.lstm import LSTMForecaster  # noqa: E402
from src.training import train as tr  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
PRED = RES / "predictions"
BASELINE_NAMES = ["persistence", "seasonal_naive_24h", "seasonal_naive_168h", "samehour_mean_7d"]
GW = 1e3


def load_all():
    comp = pd.read_csv(RES / "model_comparison.csv")
    reg = json.loads((ROOT / "models" / "registry.json").read_text())
    preds = {t: pd.read_parquet(PRED / f"{t}.parquet") for t in reg}
    return comp, reg, preds


def best_baseline(comp, task):
    g = comp[(comp.task == task) & (comp.split == "val") & comp.model.isin(BASELINE_NAMES)]
    return g.sort_values("mae").iloc[0]["model"]


def fig_comparison(comp, reg):
    tasks = list(reg)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, task in zip(axes.flat, tasks):
        g = comp[(comp.task == task) & (comp.split == "test")].set_index("model")["mae"] / GW
        v = comp[(comp.task == task) & (comp.split == "val")].set_index("model")["mae"] / GW
        order = g.sort_values().index
        x = np.arange(len(order))
        ax.bar(x - 0.2, v.reindex(order), 0.4, label="validation", color="#9ecae1")
        ax.bar(x + 0.2, g.reindex(order), 0.4, label="test", color="#08519c")
        ax.set_xticks(x)
        ax.set_xticklabels([("*" if m == reg[task]["selected_model"] else "") + m for m in order], rotation=60, ha="right", fontsize=7)
        ax.set(title=task, ylabel="MAE [GW]")
    axes[0, 0].legend()
    fig.suptitle("Model comparison (* = selected by validation MAE). FORECAST vs OBSERVED, test = 2025-01-01..06-24")
    fig.tight_layout()
    fig.savefig(FIG / "fc_model_comparison.png", dpi=110)
    plt.close(fig)


def fig_test_weeks(comp, reg, preds):
    fig, axes = plt.subplots(4, 1, figsize=(13, 13))
    for ax, tgt in zip(axes, ["solar", "demand", "wind", "net_load"]):
        task = f"{tgt}_da"
        p = preds[task]
        sel = reg[task]["selected_model"]
        bb = best_baseline(comp, task)
        wk = p.loc["2025-04-07":"2025-04-20 23:00"]
        ax.fill_between(wk.index, wk.pi_lo / GW, wk.pi_hi / GW, color="#1f77b4", alpha=0.15, label="80% PI (conformal)")
        ax.plot(wk.index, wk.y / GW, "k", lw=1.8, label="actual (OBSERVED)")
        ax.plot(wk.index, wk[sel] / GW, color="#1f77b4", lw=1.3, label=f"{sel} (selected)")
        ax.plot(wk.index, wk[bb] / GW, color="#ff7f0e", lw=1, ls="--", label=f"{bb} (best baseline)")
        ax.set(title=f"{task}: day-ahead forecast, two test weeks", ylabel="GW")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fc_test_weeks_dayahead.png", dpi=110)
    plt.close(fig)


def fig_residuals(reg, preds):
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for j, tgt in enumerate(["solar", "demand", "wind", "net_load"]):
        task = f"{tgt}_da"
        p = preds[task]
        t = p[p.split == "test"]
        sel = reg[task]["selected_model"]
        r = (t[sel] - t.y) / GW
        axes[0, j].hist(r.dropna(), bins=60, color="#08519c")
        axes[0, j].axvline(0, color="k", lw=0.8)
        axes[0, j].set(title=f"{task} residuals (test)", xlabel="forecast - actual [GW]")
        axes[1, j].scatter(t.y / GW, r, s=2, alpha=0.3)
        axes[1, j].axhline(0, color="k", lw=0.8)
        axes[1, j].set(xlabel="actual [GW]", ylabel="residual [GW]")
    fig.tight_layout()
    fig.savefig(FIG / "fc_residuals_dayahead.png", dpi=110)
    plt.close(fig)


def breakdowns(comp, reg, preds, h):
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, tgt in zip(axes, ["solar", "demand", "wind", "net_load"]):
        for hz, ls in [("da", "-"), ("h1", ":")]:
            task = f"{tgt}_{hz}"
            p = preds[task]
            t = p[p.split == "test"]
            sel = reg[task]["selected_model"]
            bb = best_baseline(comp, task)
            for name, col in [(sel, "#08519c"), (bb, "#ff7f0e")]:
                e = (t[name] - t.y).abs()
                byh = e.groupby(t.index.hour).mean() / GW
                ax.plot(byh.index, byh, ls=ls, color=col, label=f"{hz} {name}")
            # regimes
            e = (t[sel] - t.y).abs()
            eb = (t[bb] - t.y).abs()
            sol = h.loc[t.index, "solar_mw"]
            nl = h["net_load_mw"]
            ramp_day = (nl.shift(-3) - nl).loc[t.index]
            regimes = {
                "all": np.ones(len(t), bool),
                "high_solar_top10pct": (sol > sol[sol > 0].quantile(0.9)).to_numpy(),
                "evening_16_20h": np.isin(t.index.hour, [16, 17, 18, 19, 20]),
                "steep_ramp_top10pct": (ramp_day > ramp_day.quantile(0.9)).to_numpy(),
                "weekend": (t.index.dayofweek >= 5),
            }
            for rn, m in regimes.items():
                rows.append({"task": task, "regime": rn, "n": int(m.sum()), "model": sel,
                             "mae_mw": float(e[m].mean()), "baseline": bb, "baseline_mae_mw": float(eb[m].mean()),
                             "improvement_pct": float(100 * (1 - e[m].mean() / eb[m].mean()))})
            mon = e.groupby(t.index.month).mean()
            for mth, v in mon.items():
                rows.append({"task": task, "regime": f"month_{mth:02d}", "n": int((t.index.month == mth).sum()),
                             "model": sel, "mae_mw": float(v), "baseline": bb,
                             "baseline_mae_mw": float(eb[t.index.month == mth].mean()),
                             "improvement_pct": float(100 * (1 - v / eb[t.index.month == mth].mean()))})
        ax.set(title=f"{tgt}: test MAE by hour (IST)", xlabel="hour", ylabel="MAE [GW]")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fc_mae_by_hour.png", dpi=110)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "forecast_regime_breakdown.csv", index=False)

    # horizon: h1 vs da, and day-ahead MAE by lead hours (lead = 15 + hour of day)
    fig, ax = plt.subplots(figsize=(8, 4))
    hz_rows = []
    for tgt in ["solar", "demand", "wind", "net_load"]:
        p = preds[f"{tgt}_da"]
        t = p[p.split == "test"]
        e = (t[reg[f"{tgt}_da"]["selected_model"]] - t.y).abs()
        nmae = e.groupby(t.index.hour).mean() / t.y.abs().mean() * 100
        ax.plot(15 + nmae.index, nmae, marker=".", label=tgt)
        for hz in ["h1", "da"]:
            g = comp[(comp.task == f"{tgt}_{hz}") & (comp.split == "test") & (comp.model == reg[f"{tgt}_{hz}"]["selected_model"])].iloc[0]
            hz_rows.append({"target": tgt, "horizon": hz, "lead_hours": "1" if hz == "h1" else "15-38",
                            "model": g.model, "mae": g.mae, "rmse": g.rmse, "r2": g.r2, "nmae_pct": g.nmae_pct})
    ax.set(xlabel="lead time from issue (h)", ylabel="nMAE % (of mean actual)", title="Day-ahead test error vs lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fc_error_by_lead.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(hz_rows).to_csv(RES / "forecast_by_horizon.csv", index=False)
    return df


def importance(reg, h):
    out = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, tgt in zip(axes, ["solar_mw", "demand_mw", "wind_mw", "net_load_mw"]):
        task = f"{tgt.replace('_mw', '')}_da"
        name = reg[task]["selected_model"]
        if name in BASELINE_NAMES or name == "lstm":
            name = "lightgbm"  # interpret the strongest tabular model instead
        art = joblib.load(ROOT / "models" / task / f"{name}.joblib")
        spec = default_spec(tgt, "da")
        X, y, _ = build_features(h, spec)
        m = split_mask(X.index, "test").to_numpy()
        Xt, yt = X[m][art["features"]], y[m]
        pi = permutation_importance(art["model"], Xt, yt, scoring="neg_mean_absolute_error", n_repeats=3,
                                    random_state=0, n_jobs=1)
        s = pd.Series(pi.importances_mean, index=Xt.columns).sort_values(ascending=False)
        for f, v in s.items():
            out.append({"task": task, "model": name, "feature": f, "mae_increase_mw": float(v)})
        s.head(12)[::-1].plot.barh(ax=ax, color="#08519c")
        ax.set(title=f"{task} ({name}): permutation importance (test)", xlabel="MAE increase [MW]")
    fig.tight_layout()
    fig.savefig(FIG / "fc_permutation_importance.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(out).to_csv(RES / "feature_importance.csv", index=False)


def experiments(h):
    wx = load_weather()
    rows = []
    for tgt in ["solar_mw", "demand_mw", "net_load_mw"]:
        for variant in ["full", "with_weather", "no_lags", "calendar_only"]:
            spec = default_spec(tgt, "da")
            w = None
            if variant == "with_weather":
                if wx is None:
                    continue
                w = wx
                spec.weather = list(wx.columns)
            elif variant == "no_lags":
                spec.target_lags, spec.exog_lags = [], {}
            elif variant == "calendar_only":
                spec.target_lags, spec.exog_lags, spec.issue_features = [], {}, False
            r = tr.run_task(h, tgt, "da", weather=w, families=["lightgbm"], run_lstm=False, save=False,
                            spec=spec, log=lambda s: None)
            c = r["comparison"]
            for part in ["val", "test"]:
                g = c[(c.model == "lightgbm") & (c.split == part)].iloc[0]
                rows.append({"target": tgt, "horizon": "da", "variant": variant, "split": part, "mae": g.mae,
                             "rmse": g.rmse, "r2": g.r2, "nmae_pct": g.nmae_pct,
                             "note": "weather = archived short-lead NWP (optimistic for day-ahead)" if variant == "with_weather" else ""})
            print(f"  E4 {tgt} {variant}: test MAE {rows[-1]['mae']:,.0f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "feature_ablation_weather.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-experiments", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    h = load_hourly()
    comp, reg, preds = load_all()
    fig_comparison(comp, reg)
    fig_test_weeks(comp, reg, preds)
    fig_residuals(reg, preds)
    br = breakdowns(comp, reg, preds, h)
    print(br[br.regime.isin(["all", "high_solar_top10pct", "evening_16_20h", "steep_ramp_top10pct"])].round(1).to_string())
    importance(reg, h)
    if not args.skip_experiments:
        print(experiments(h).round(2).to_string())


if __name__ == "__main__":
    main()
