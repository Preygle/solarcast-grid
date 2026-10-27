"""Open-Meteo archived NWP forecasts (Historical Forecast API) for the weather experiment.

Variables are short-lead model forecasts stitched by Open-Meteo, so for day-ahead use they are
OPTIMISTIC relative to what an operator would have had at 10:00 on day D. They are only used in
the separately labelled weather experiment (E4), never in the main feature set.

Sites: large solar parks (radiation, cloud) and large load centres (temperature).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
WEATHER_RAW = ROOT / "data" / "raw" / "open_meteo"
WEATHER_PATH = ROOT / "data" / "processed" / "weather_hourly.parquet"
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"

SOLAR_SITES = {  # approximate coordinates of major solar parks
    "bhadla_rj": (27.54, 71.92), "pavagada_ka": (14.25, 77.28), "charanka_gj": (23.90, 71.20),
    "kurnool_ap": (15.68, 78.10), "rewa_mp": (24.53, 81.30), "kamuthi_tn": (9.35, 78.38),
}
LOAD_SITES = {
    "delhi": (28.61, 77.21), "mumbai": (19.08, 72.88), "chennai": (13.08, 80.27),
    "kolkata": (22.57, 88.36), "lucknow": (26.85, 80.95),
}
START, END = "2021-09-01", "2025-06-24"


def fetch_site(name: str, lat: float, lon: float, variables: list[str]) -> pd.DataFrame:
    WEATHER_RAW.mkdir(parents=True, exist_ok=True)
    cache = WEATHER_RAW / f"{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    r = requests.get(API, params={
        "latitude": lat, "longitude": lon, "start_date": START, "end_date": END,
        "hourly": ",".join(variables), "timezone": "Asia/Kolkata"}, timeout=120)
    r.raise_for_status()
    js = r.json()["hourly"]
    df = pd.DataFrame(js).set_index("time")
    df.index = pd.to_datetime(df.index)
    df.to_csv(cache)
    return df


def build_weather() -> pd.DataFrame:
    """Hourly IST features. Open-Meteo radiation is the mean over the *preceding* hour, labelled at
    hour end, so it is shifted by -1 h to match our start-of-hour labels."""
    out = {}
    rad, cloud = [], []
    for n, (la, lo) in SOLAR_SITES.items():
        d = fetch_site(n, la, lo, ["shortwave_radiation", "cloud_cover"])
        rad.append(d["shortwave_radiation"].shift(-1).rename(n))
        cloud.append(d["cloud_cover"].rename(n))
    rad, cloud = pd.concat(rad, axis=1), pd.concat(cloud, axis=1)
    out["wx_rad_mean"] = rad.mean(axis=1)
    # All-India solar peaks ~30 min later than the site-mean radiation; the next hour's forecast
    # radiation (same NWP run, same target day) lets models interpolate.
    out["wx_rad_mean_next"] = out["wx_rad_mean"].shift(-1)
    out["wx_rad_min"] = rad.min(axis=1)
    out["wx_cloud_mean"] = cloud.mean(axis=1)
    temps = []
    for n, (la, lo) in LOAD_SITES.items():
        d = fetch_site(n, la, lo, ["temperature_2m"])
        temps.append(d["temperature_2m"].rename(n))
    temps = pd.concat(temps, axis=1)
    out["wx_temp_mean"] = temps.mean(axis=1)
    out["wx_temp_max"] = temps.max(axis=1)
    out["wx_temp_north"] = temps[["delhi", "lucknow"]].mean(axis=1)
    w = pd.DataFrame(out)
    w.index.name = "timestamp"
    WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(WEATHER_PATH)
    return w


def load_weather() -> pd.DataFrame | None:
    return pd.read_parquet(WEATHER_PATH) if WEATHER_PATH.exists() else None
