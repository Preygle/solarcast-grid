"""Phase 2 analysis: summary tables, EDA figures, duck-curve metrics, surplus proxy.

Usage: python scripts/run_eda.py   (requires data/processed/hourly.parquet)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.analysis import duck_curve as dc  # noqa: E402
from src.analysis import surplus as sp  # noqa: E402
from src.analysis.eda import plot_all, summary_table  # noqa: E402
from src.data.preprocess import load_hourly  # noqa: E402

OUT = ROOT / "results" / "eda"


def main() -> None:
    h = load_hourly()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_table(h).to_csv(OUT / "summary_statistics.csv")

    met = dc.daily_duck_metrics(h)
    met.to_csv(OUT / "daily_duck_metrics.csv")
    yearly = met.groupby(met.index.year).agg(
        days=("net_load_min_mw", "size"),
        net_load_min_mw=("net_load_min_mw", "min"),
        mean_daily_min_mw=("net_load_min_mw", "mean"),
        net_load_max_mw=("net_load_max_mw", "max"),
        mean_evening_ramp_3h_mw=("evening_ramp_3h_mw", "mean"),
        p95_evening_ramp_3h_mw=("evening_ramp_3h_mw", lambda s: s.quantile(0.95)),
        max_evening_ramp_3h_mw=("evening_ramp_3h_mw", "max"),
        duck_shape_days_pct=("duck_shape", lambda s: 100 * s.mean()),
        mean_solar_share_pct=("solar_share_pct", "mean"),
        modal_min_hour=("net_load_min_hour", lambda s: int(s.mode().iloc[0])),
    )
    yearly.to_csv(OUT / "duck_yearly_summary.csv")

    seasonal =met.assign(season=met.index.month.map(dc.SEASONS), year=met.index.year).groupby(["year", "season"]).agg(
        duck_shape_days_pct=("duck_shape", lambda s: 100 * s.mean()),
        mean_evening_ramp_3h_mw=("evening_ramp_3h_mw", "mean"),
        mean_daily_min_mw=("net_load_min_mw", "mean"))
    seasonal.to_csv(OUT / "duck_seasonal_summary.csv")

    tab = sp.scenario_table(h)
    tab.to_csv(OUT / "surplus_proxy_scenarios.csv", index=False)
    sanity = {f"theta_{t}": sp.thermal_sanity(h, t) for t in sp.THETAS}
    (OUT / "surplus_proxy_sanity.json").write_text(json.dumps(
        {"fraction_hours_observed_thermal_below_floor": sanity}, indent=2))

    figs = plot_all(h, OUT)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(yearly.round(1).to_string())
        print(seasonal.round(1).to_string())
        print(tab.round(2).to_string())
        print("sanity", sanity)
    print("figures:", figs)


if __name__ == "__main__":
    main()
