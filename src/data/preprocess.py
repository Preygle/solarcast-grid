"""Clean raw SCADA data and build the hourly analysis table.

Steps (each counted in the preprocessing log):
1. Dropout masking: rows where demand == 0 are telemetry dropouts -> all values NaN.
   Exact zeros in non-solar columns are also treated as dropouts.
2. Physical-range masking: values outside ``BOUNDS`` -> NaN.
3. Hampel spike filter on the raw-resolution series: |x - rolling median| >
   max(K * 1.4826 * MAD, floor) -> NaN.
4. Solar: negative night-time metering offsets clipped to 0 (after steps 1-3).
5. Hourly mean (timestamp = start of hour, IST). An hour is kept only if
   >= MIN_COVERAGE of its samples are valid; otherwise NaN.
6. Frozen telemetry: >= FLATLINE_H identical consecutive hourly values -> NaN.
7. Solar seasonal Hampel: deviation from the same clock-hour median over a
   15-day window -> NaN (catches stuck values such as 4.8 GW at 22:00).
8. Solar night offset: the day's median 00:00-03:59 value is subtracted
   (metering offset); NIGHT_HOURS are set to 0 MW.
9. Short gaps (<= MAX_INTERP_H hours) linearly interpolated and flagged in
   ``<col>_imputed``; longer gaps stay NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import RAW_DIR, month_files, read_hourly_2024, read_month

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
HOURLY_PATH = PROCESSED / "hourly.parquet"
LOG_PATH = ROOT / "results" / "eda" / "preprocessing_log.json"

VALUE_COLS = ["demand_mw", "solar_raw_mw", "wind_mw", "thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw"]

# Plausible all-India ranges [MW] for 2021-2025. They are set well outside observed normal
# operation (see DATA_AUDIT.md) and only reject impossible values.
BOUNDS = {
    "demand_mw": (80_000, 260_000),
    "solar_raw_mw": (-3_000, 80_000),
    "wind_mw": (0, 35_000),
    "thermal_mw": (40_000, 200_000),
    "hydro_mw": (1_000, 65_000),
    "gas_mw": (0, 15_000),
    "nuclear_mw": (1_000, 9_000),
}
# Minimum Hampel deviation [MW] so that flat periods (zero MAD) do not flag noise.
HAMPEL_FLOOR = {
    "demand_mw": 4_000, "solar_raw_mw": 2_000, "wind_mw": 1_500, "thermal_mw": 4_000,
    "hydro_mw": 2_000, "gas_mw": 800, "nuclear_mw": 400,
}
HAMPEL_K = 6.0
HAMPEL_WINDOW = "1h"
MIN_COVERAGE = 0.5
MAX_INTERP_H = 2
# Hours (start, IST) with no sunlight anywhere in India in any season.
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3]
SEASONAL_WINDOW_DAYS = 15
SEASONAL_FLOOR_MW = 1_500


FLATLINE_H = 3
HOURLY_HAMPEL_FLOOR = {"demand_mw": 12_000, "thermal_mw": 12_000, "wind_mw": 6_000}


def flatline_mask(s: pd.Series, min_run: int = FLATLINE_H) -> pd.Series:
    """Flag runs of >= min_run exactly-equal consecutive non-zero values (frozen telemetry)."""
    same = s.eq(s.shift()) & s.ne(0) & s.notna()
    start = ~same
    run_id = start.cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (run_len >= min_run) & s.ne(0) & s.notna()


def seasonal_hampel(s: pd.Series, k: float = HAMPEL_K, days: int = SEASONAL_WINDOW_DAYS,
                    floor: float = SEASONAL_FLOOR_MW) -> pd.Series:
    """Flag hourly values far from the median of the same clock-hour over a centred window of days."""
    flags = pd.Series(False, index=s.index)
    for _, g in s.groupby(s.index.hour):
        med = g.rolling(days, center=True, min_periods=5).median()
        mad = (g - med).abs().rolling(days, center=True, min_periods=5).median()
        thresh = np.maximum(k * 1.4826 * mad, floor)
        flags.loc[g.index] = ((g - med).abs() > thresh) & g.notna() & med.notna()
    return flags


def _hampel(s: pd.Series, window: str, floor: float) -> pd.Series:
    """Boolean mask of spike outliers in ``s`` (time-indexed)."""
    med = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=3).median()
    thresh = np.maximum(HAMPEL_K * 1.4826 * mad, floor)
    return ((s - med).abs() > thresh) & s.notna() & med.notna()


def clean_frame(df: pd.DataFrame, window: str = HAMPEL_WINDOW) -> tuple[pd.DataFrame, dict]:
    """Apply steps 1-4 to one raw frame. Returns the cleaned frame and per-step counts."""
    df = df.copy()
    cols = [c for c in VALUE_COLS if c in df]
    log: dict = {"rows": int(len(df))}

    dropout_rows = df["demand_mw"] == 0
    log["dropout_rows_demand_zero"] = int(dropout_rows.sum())
    df.loc[dropout_rows, cols] = np.nan
    for c in cols:
        if c != "solar_raw_mw":
            z = df[c] == 0
            log[f"{c}_zero_masked"] = int(z.sum())
            df.loc[z, c] = np.nan
        lo, hi = BOUNDS[c]
        out = (df[c] < lo) | (df[c] > hi)
        log[f"{c}_out_of_bounds"] = int(out.sum())
        df.loc[out, c] = np.nan
        spikes = _hampel(df[c], window, HAMPEL_FLOOR[c])
        log[f"{c}_hampel_spikes"] = int(spikes.sum())
        df.loc[spikes, c] = np.nan

    neg = df["solar_raw_mw"] < 0
    log["solar_negative_clipped"] = int(neg.sum())
    df["solar_mw"] = df["solar_raw_mw"].clip(lower=0)
    return df.drop(columns=["solar_raw_mw"]), log


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean with minimum-coverage rule (step 5)."""
    cols = [c for c in df.columns if c.endswith("_mw")]
    g = df[cols].resample("1h")
    mean, valid, total = g.mean(), g.count(), df[cols[0]].resample("1h").size()
    coverage = valid.div(total.replace(0, np.nan), axis=0)
    return mean.where(coverage >= MIN_COVERAGE)


def build_hourly(use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline and return (hourly frame, preprocessing log)."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    logs: dict = {"files": {}, "params": {
        "bounds": BOUNDS, "hampel_floor": HAMPEL_FLOOR, "hampel_k": HAMPEL_K,
        "hampel_window_raw": HAMPEL_WINDOW, "hampel_window_hourly_file": "5h",
        "min_coverage": MIN_COVERAGE, "max_interp_h": MAX_INTERP_H}}
    parts = []
    for p in month_files(RAW_DIR):
        cache = INTERIM / f"{p.stem}.parquet"
        log_cache = INTERIM / f"{p.stem}.json"
        if use_cache and cache.exists() and log_cache.exists():
            hourly, log = pd.read_parquet(cache), json.loads(log_cache.read_text())
        else:
            clean, log = clean_frame(read_month(p))
            hourly = to_hourly(clean)
            hourly.to_parquet(cache)
            log_cache.write_text(json.dumps(log))
        logs["files"][p.name] = log
        hourly["source_file"] = p.name
        parts.append(hourly)

    h24 = read_hourly_2024(RAW_DIR)
    h24_total = h24.pop("total_generation_mw")
    clean24, log24 = clean_frame(h24.drop(columns=["source_file"]), window="5h")
    log24["note"] = "hourly source; aggregation method (snapshot vs mean) undocumented"
    logs["files"][h24["source_file"].iloc[0]] = log24
    clean24["total_generation_mw"] = h24_total
    clean24["source_file"] = h24["source_file"].iloc[0]
    parts.append(clean24)

    hourly = pd.concat(parts).sort_index()
    if hourly.index.duplicated().any():
        raise ValueError("duplicate hourly timestamps after concatenation")
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h", name="timestamp")
    hourly = hourly.reindex(full)
    hourly["source_file"] = hourly["source_file"].ffill()

    # Step 6b: hour-level spikes that survived the raw-resolution filter (partial-hour dropouts
    # pull an hourly mean far below its neighbours, e.g. 155 -> 107 -> 154 GW demand).
    logs["hourly_hampel_masked"] = {}
    for c, floor in HOURLY_HAMPEL_FLOOR.items():
        spikes = _hampel(hourly[c], "5h", floor)
        logs["hourly_hampel_masked"][c] = int(spikes.sum())
        hourly.loc[spikes, c] = np.nan

    # Sustained multi-hour dropouts (e.g. demand 105 GW for 6 h on a ~180 GW day): same clock-hour
    # check with a 25 GW floor, far above holiday effects (~10-20 GW).
    dflags = seasonal_hampel(hourly["demand_mw"], floor=25_000)
    hourly.loc[dflags, "demand_mw"] = np.nan
    logs["hourly_hampel_masked"]["demand_mw_seasonal"] = int(dflags.sum())

    # Step 7: frozen telemetry -- the same value repeated exactly for >= FLATLINE_H hours.
    logs["flatline_masked"] = {}
    for c in ["demand_mw", "solar_mw", "wind_mw", "thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw"]:
        flat = flatline_mask(hourly[c])
        logs["flatline_masked"][c] = int(flat.sum())
        hourly.loc[flat, c] = np.nan

    # Step 8 (solar): stuck values far above the same clock-hour on neighbouring days.
    flags = seasonal_hampel(hourly["solar_mw"])
    hourly.loc[flags, "solar_mw"] = np.nan
    logs["solar_seasonal_hampel_flags"] = int(flags.sum())

    # Step 9 (solar): per-day metering offset = median of 00:00-03:59 values; subtracted from
    # that day's daylight hours, and night hours set to 0 (no sunlight anywhere in India).
    early = hourly["solar_mw"].where(hourly.index.hour.isin([0, 1, 2, 3]))
    offset = early.groupby(hourly.index.date).transform("median").fillna(0).clip(lower=0)
    night = hourly.index.hour.isin(NIGHT_HOURS)
    hourly["solar_mw"] = (hourly["solar_mw"] - offset).clip(lower=0)
    night_fix = night
    hourly.loc[night_fix, "solar_mw"] = 0.0
    logs["solar_offset_mw_mean"] = float(offset.mean())
    logs["solar_offset_mw_max"] = float(offset.max())
    logs["solar_night_set_zero"] = int(night_fix.sum())

    interp_cols = ["demand_mw", "solar_mw", "wind_mw", "thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw"]
    logs["hourly"] = {}
    for c in interp_cols:
        before = hourly[c].isna()
        filled = hourly[c].interpolate(method="time", limit=MAX_INTERP_H, limit_area="inside")
        # Only fill gaps whose total length is <= MAX_INTERP_H.
        run_id = before.ne(before.shift()).cumsum()
        run_len = before.groupby(run_id).transform("sum")
        fill_mask = before & (run_len <= MAX_INTERP_H)
        hourly[c] = hourly[c].where(~fill_mask, filled)
        hourly[f"{c}_imputed"] = fill_mask & hourly[c].notna()
        logs["hourly"][c] = {"nan_before_interp": int(before.sum()),
                             "imputed": int(hourly[f'{c}_imputed'].sum()),
                             "nan_after": int(hourly[c].isna().sum())}
    # The mix columns do not exist after 2023 by construction; record that separately.
    post = hourly.index >= "2024-01-01"
    for c in ["thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw"]:
        logs["hourly"][c]["nan_after_pre2024"] = int(hourly.loc[~post, c].isna().sum())

    hourly["net_load_mw"] = hourly["demand_mw"] - hourly["solar_mw"] - hourly["wind_mw"]
    logs["hourly_rows"] = int(len(hourly))
    logs["start"], logs["end"] = str(hourly.index.min()), str(hourly.index.max())
    return hourly, logs


def run(use_cache: bool = True) -> pd.DataFrame:
    hourly, logs = build_hourly(use_cache=use_cache)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_PATH)
    LOG_PATH.write_text(json.dumps(logs, indent=2, default=str))
    return hourly


def load_hourly() -> pd.DataFrame:
    if not HOURLY_PATH.exists():
        raise FileNotFoundError(f"{HOURLY_PATH} missing; run `python scripts/build_dataset.py`")
    return pd.read_parquet(HOURLY_PATH)
