"""Audit every raw workbook: resolution, coverage, duplicates, missing values, ranges.

Writes results/eda/raw_audit.csv and prints a summary. Usage: python scripts/audit_raw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_raw import month_files, read_hourly_2024, read_month  # noqa: E402

VALUE_COLS = ["thermal_mw", "hydro_mw", "gas_mw", "nuclear_mw", "wind_mw", "solar_raw_mw", "demand_mw"]


def audit(df: pd.DataFrame, name: str) -> dict:
    idx = df.index
    step = pd.Series(idx).diff().dropna()
    cols = [c for c in VALUE_COLS if c in df]
    row = {
        "file": name,
        "rows": len(df),
        "start": idx.min(),
        "end": idx.max(),
        "modal_step_s": step.mode().iloc[0].total_seconds() if len(step) else None,
        "n_distinct_steps": step.nunique(),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "non_monotonic": int((step <= pd.Timedelta(0)).sum()),
        "max_gap_min": step.max().total_seconds() / 60 if len(step) else None,
    }
    for c in cols:
        row[f"{c}_nan"] = int(df[c].isna().sum())
        row[f"{c}_min"] = float(df[c].min())
        row[f"{c}_max"] = float(df[c].max())
    return row


def main() -> None:
    rows = []
    for p in month_files():
        rows.append(audit(read_month(p), p.name))
        print(rows[-1]["file"], rows[-1]["rows"], rows[-1]["modal_step_s"], rows[-1]["duplicate_timestamps"])
    h = read_hourly_2024()
    rows.append(audit(h, h["source_file"].iloc[0]))
    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "eda"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "raw_audit.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(out.to_string())


if __name__ == "__main__":
    main()
