"""Exploratory plots and tables for the hourly Grid-India data. All figures are tagged
OBSERVED or DERIVED in their titles."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import duck_curve as dc  # noqa: E402
from src.analysis import surplus as sp  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
GW = 1e3
COLORS = {"demand_mw": "#1f4e79", "solar_mw": "#e69f00", "wind_mw": "#009e73", "net_load_mw": "#7a1f5c"}
SEASON_ORDER = ["Winter (DJF)", "Pre-monsoon (MAM)", "Monsoon (JJAS)", "Post-monsoon (ON)"]


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def summary_table(h: pd.DataFrame) -> pd.DataFrame:
    cols = ["demand_mw", "solar_mw", "wind_mw", "net_load_mw", "thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw"]
    t = h[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
    t["missing"] = h[cols].isna().sum()
    t["imputed"] = [int(h[f"{c}_imputed"].sum()) if f"{c}_imputed" in h else 0 for c in cols]
    return t


def plot_all(h: pd.DataFrame, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1. Daily means over time
    d = h[["demand_mw", "solar_mw", "wind_mw", "net_load_mw"]].resample("D").mean() / GW
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in d:
        ax.plot(d.index, d[c], lw=0.9, color=COLORS[c], label=c.replace("_mw", ""))
    ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls="--", lw=0.8)
    ax.text(pd.Timestamp("2024-01-10"), ax.get_ylim()[1] * 0.95, "source switches to hourly report", fontsize=8, color="grey")
    ax.set(title="All-India daily mean power (OBSERVED; net load DERIVED)", ylabel="GW")
    ax.legend(ncol=4, loc="center left")
    figs.append(_save(fig, out, "01_daily_mean_timeseries"))

    # 2. Hour-of-day profiles by year
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, c in zip(axes, ["solar_mw", "demand_mw", "net_load_mw"]):
        prof = h.groupby([h.index.year, h.index.hour])[c].mean().unstack(0) / GW
        prof.plot(ax=ax, lw=1.6, cmap="viridis")
        ax.set(title=f"Mean {c.replace('_mw', '')} by hour (IST)", xlabel="hour", ylabel="GW")
        ax.legend(title="year", fontsize=8)
    figs.append(_save(fig, out, "02_hourly_profiles_by_year"))

    # 3. Day-of-week effect on demand (deviation from weekly mean, by year)
    dd = h["demand_mw"].resample("D").mean()
    rel = dd / dd.rolling(7, center=True).mean() - 1
    dow = rel.groupby(rel.index.dayofweek).agg(["mean", "std"]) * 100
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow["mean"], yerr=dow["std"], color=COLORS["demand_mw"], alpha=0.8)
    ax.set(title="Day-of-week effect on daily demand (DERIVED)", ylabel="% vs centred 7-day mean")
    figs.append(_save(fig, out, "03_day_of_week_demand"))

    # 4. Monthly distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, c in zip(axes, ["demand_mw", "solar_mw"]):
        m = [h.loc[h.index.month == i, c].dropna() / GW for i in range(1, 13)]
        if c == "solar_mw":
            m = [s[s > 0.5] for s in m]
        ax.boxplot(m, showfliers=False)
        ax.set(title=f"Monthly distribution of hourly {c.replace('_mw', '')}" + (" (daylight)" if c == "solar_mw" else ""),
               xlabel="month", ylabel="GW")
    figs.append(_save(fig, out, "04_monthly_boxplots"))

    # 5. Solar vs demand
    s = h[["solar_mw", "demand_mw"]].dropna()
    s = s[s.index.year >= 2024]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(s["solar_mw"] / GW, s["demand_mw"] / GW, c=s.index.hour, cmap="twilight", s=3, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="hour")
    ax.set(title="Solar vs demand, 2024-25 hourly (OBSERVED)", xlabel="solar GW", ylabel="demand GW")
    figs.append(_save(fig, out, "05_solar_vs_demand"))

    # 6. Missing / imputed pattern
    miss = pd.DataFrame({c: h[c].isna().resample("ME").sum() for c in ["demand_mw", "solar_mw", "wind_mw"]})
    imp = pd.DataFrame({c: h[f"{c}_imputed"].resample("ME").sum() for c in ["demand_mw", "solar_mw", "wind_mw"]})
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    miss.plot.bar(ax=axes[0], width=0.85); imp.plot.bar(ax=axes[1], width=0.85)
    for ax, t in zip(axes, ["hours still missing", "hours interpolated (gap <= 2 h)"]):
        ax.set(title=t, xlabel="")
        ax.set_xticks(range(0, len(miss), 3)); ax.set_xticklabels([i.strftime("%Y-%m") for i in miss.index[::3]], rotation=60, fontsize=7)
    figs.append(_save(fig, out, "06_missing_imputed_by_month"))

    # 7. Duck curve: seasonal net-load profiles per year
    prof = dc.seasonal_profiles(h)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, season in zip(axes, SEASON_ORDER):
        sub = prof.xs(season, level="season")
        for yr, row in sub.iterrows():
            ax.plot(row.index, row.values / GW, lw=1.6, label=str(yr), color=plt.cm.viridis((yr - 2021) / 4))
        ax.set(title=season, xlabel="hour (IST)")
    axes[0].set_ylabel("mean net load, GW")
    handles = {lab: hd for ax in axes for hd, lab in zip(*ax.get_legend_handles_labels())}
    axes[-1].legend([handles[k] for k in sorted(handles)], sorted(handles), fontsize=8)
    fig.suptitle("Net load = demand - solar - wind: seasonal mean profiles (DERIVED)")
    figs.append(_save(fig, out, "07_duck_curve_seasonal"))

    # 8. Demand vs net load on the lowest-midday-net-load day per year
    met = dc.daily_duck_metrics(h)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=True)
    for ax, (yr, g) in zip(axes, met.groupby(met.index.year)):
        # Deep-duck but not extreme: the day at the 10th percentile of daily minimum net load.
        q = g["net_load_min_mw"].quantile(0.10)
        day = (g["net_load_min_mw"] - q).abs().idxmin()
        x = h.loc[day: day + pd.Timedelta(hours=23)]
        for c in ["demand_mw", "net_load_mw", "solar_mw"]:
            ax.plot(x.index.hour, x[c] / GW, color=COLORS[c], lw=1.8, label=c.replace("_mw", ""))
        ax.set(title=f"{day:%Y-%m-%d}", xlabel="hour")
    axes[0].set_ylabel("GW"); axes[0].legend(fontsize=8)
    fig.suptitle("Representative deep-duck days: 10th-percentile daily minimum net load in each year (OBSERVED/DERIVED)")
    figs.append(_save(fig, out, "08_duck_representative_days"))

    # 9. Evening ramp
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    r = met["evening_ramp_3h_mw"] / GW
    axes[0].plot(r.index, r, lw=0.5, alpha=0.5, color="grey")
    axes[0].plot(r.rolling(30, center=True).mean(), color=COLORS["net_load_mw"], lw=2, label="30-day mean")
    axes[0].set(title="Max 3-h evening net-load ramp (start 15-18 h)", ylabel="GW / 3 h"); axes[0].legend()
    mon = met.groupby([met.index.year, met.index.month])["evening_ramp_3h_mw"].mean().unstack(0) / GW
    mon.plot(ax=axes[1], marker="o", cmap="viridis")
    axes[1].set(title="Monthly mean evening ramp by year", xlabel="month", ylabel="GW / 3 h")
    figs.append(_save(fig, out, "09_evening_ramp"))

    # 10. Surplus proxy
    tab = sp.scenario_table(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    for theta, g in tab.groupby("theta"):
        ax.plot(g["solar_scale_k"], g["surplus_share_of_solar_pct"], marker="o", label=f"theta={theta}")
    ax.set(title="Surplus-risk PROXY: share of (scaled) solar energy in surplus\n2021-09..2023-12 mix, assumption-driven",
           xlabel="solar scale-up factor k", ylabel="% of solar energy")
    ax.legend()
    figs.append(_save(fig, out, "10_surplus_proxy_scenarios"))

    hr = sp.headroom(h, 0.55, 2.0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(hr.groupby(hr.index.hour).quantile(0.05) / GW, label="5th pct")
    ax.plot(hr.groupby(hr.index.hour).median() / GW, label="median")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Headroom = demand - (2x solar + wind + floor), theta=0.55\n(<0 = surplus under proxy)", xlabel="hour", ylabel="GW")
    ax.legend()
    figs.append(_save(fig, out, "11_surplus_headroom_by_hour"))
    return figs
