"""Build data/processed/hourly.parquet from the raw workbooks.

Usage: python scripts/build_dataset.py [--no-cache]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import HOURLY_PATH, LOG_PATH, run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="re-parse raw xlsx instead of data/interim cache")
    args = ap.parse_args()
    t = time.time()
    df = run(use_cache=not args.no_cache)
    print(f"wrote {HOURLY_PATH} {df.shape} in {time.time() - t:.0f}s; log -> {LOG_PATH}")
    print(df.describe().T.to_string())
