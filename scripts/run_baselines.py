"""Phase 2 baseline forecasts on the chronological validation and test splits.

Usage: python scripts/run_baselines.py   -> results/baseline_metrics.csv, results/eda/12_baselines_test_week.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import load_hourly  # noqa: E402
from src.evaluation.metrics import mase_scale, regression_metrics  # noqa: E402
from src.features.splits import split  # noqa: E402
from src.models.baselines import BASELINES  # noqa: E402

TARGETS = ["solar_mw", "demand_mw", "wind_mw", "net_load_mw"]
HORIZONS = ["h1", "da"]


def main() -> None:
    h = load_hourly()
    rows = []
    for tgt in TARGETS:
        y = h[tgt]
        scale = mase_scale(split(y, "train"))
        for hz in HORIZONS:
            for name, fn in BASELINES.items():
                p = fn(y, hz)
                for part in ["val", "test"]:
                    m = regression_metrics(split(y, part), split(p, part), tgt, scale)
                    rows.append({"target": tgt, "horizon": hz, "model": name, "split": part, **m})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "baseline_metrics.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out[out.split == "test"][["target", "horizon", "model", "mae", "rmse", "r2", "nmae_pct", "mase"]].round(3).to_string(index=False))

    wk = h.loc["2025-04-07":"2025-04-13"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, tgt in zip(axes, ["solar_mw", "demand_mw"]):
        ax.plot(wk.index, wk[tgt] / 1e3, color="k", lw=2, label="actual (OBSERVED)")
        for name in ["persistence", "seasonal_naive_24h", "samehour_mean_7d"]:
            p = BASELINES[name](h[tgt], "da").loc[wk.index]
            ax.plot(wk.index, p / 1e3, lw=1, label=f"{name} (day-ahead)")
        ax.set(ylabel="GW", title=f"{tgt}: test week, day-ahead baselines (FORECAST)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "eda" / "12_baselines_test_week.png")


if __name__ == "__main__":
    main()
