"""Parse the Grid-India SCADA spreadsheets from Mendeley dataset y58jknpgs8.

Monthly workbooks (Sep 2021 - Dec 2023) have a ``Sheet1`` with two header rows:
row 0 = SCADA point id, row 1 = point name (e.g. ``ALL_IND_SOLAR|P``), then a
``Time`` column followed by values in MW. Sampling is 5 min (Sep 2021 - Sep 2022)
or 10 s (Oct 2022 onward, see DATA_AUDIT.md).

The combined ``January 2024- June 2025.xlsx`` workbook is an hourly ``Report``
sheet with Demand, Wind, Solar and Total Generation only, timestamps in
DD-MM-YYYY format, and three trailing summary rows (Max timestamp / Average / Sum).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "mendeley_y58jknpgs8"
HOURLY_2024_FILE = "January 2024- June 2025.xlsx"

# SCADA point name -> canonical column (all MW, all-India aggregates).
POINT_MAP = {
    "TOTAL_THM_ONLY|P": "thermal_mw",
    "TOTAL_NLDC_HYD|P": "hydro_mw",
    "TOTAL_NLDC_GAS|P": "gas_mw",
    "TOTAL_NLDC_NPC|P": "nuclear_mw",
    "ALL_INDIA_WIND|P": "wind_mw",
    "ALL_IND_SOLAR|P": "solar_raw_mw",
    "NLDC_DEMAND|P": "demand_mw",
}
CANONICAL = list(POINT_MAP.values())

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def month_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Monthly workbooks ("<Month> <Year>.xlsx"), sorted chronologically."""
    out = []
    for p in raw_dir.glob("*.xlsx"):
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", p.stem.strip())
        if not m:
            continue
        name = m.group(1).lower().replace("novemeber", "november")  # typo in source file name
        if name in MONTHS:
            out.append((int(m.group(2)), MONTHS[name], p))
    return [p for _, _, p in sorted(out)]


def read_month(path: Path) -> pd.DataFrame:
    """Read one monthly SCADA workbook into a frame indexed by timestamp (raw resolution)."""
    df = pd.read_excel(path, sheet_name="Sheet1", header=None, engine="calamine")
    names = df.iloc[1].tolist()
    cols = {j: POINT_MAP[n.strip()] for j, n in enumerate(names)
            if isinstance(n, str) and n.strip() in POINT_MAP}
    missing = set(CANONICAL) - set(cols.values())
    if missing:
        raise ValueError(f"{path.name}: missing SCADA points {sorted(missing)}")
    body = df.iloc[2:]
    out = pd.DataFrame({c: pd.to_numeric(body[j], errors="coerce").to_numpy() for j, c in cols.items()},
                       index=pd.to_datetime(body[0], errors="coerce").to_numpy())
    out.index.name = "timestamp"
    out = out[out.index.notna()]
    out["source_file"] = path.name
    return out


def read_hourly_2024(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read the hourly Jan 2024 - Jun 2025 report, dropping the trailing summary rows."""
    df = pd.read_excel(raw_dir / HOURLY_2024_FILE, sheet_name="Report", engine="calamine")
    ts = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df = df[ts.notna()].copy()
    out = pd.DataFrame({
        "demand_mw": pd.to_numeric(df["Demand (MW)"], errors="coerce").to_numpy(),
        "wind_mw": pd.to_numeric(df["Wind (MW)"], errors="coerce").to_numpy(),
        "solar_raw_mw": pd.to_numeric(df["Solar (MW)"], errors="coerce").to_numpy(),
        "total_generation_mw": pd.to_numeric(df["Total Generation (MW)"], errors="coerce").to_numpy(),
    }, index=ts[ts.notna()].to_numpy())
    out.index.name = "timestamp"
    out["source_file"] = HOURLY_2024_FILE
    return out
