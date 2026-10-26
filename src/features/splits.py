"""Chronological train / validation / test split (PROJECT_SPEC §4)."""
from __future__ import annotations

import pandas as pd

SPLITS = {
    "train": ("2021-09-01", "2024-06-30 23:00"),
    "val": ("2024-07-01", "2024-12-31 23:00"),
    "test": ("2025-01-01", "2025-06-24 23:00"),
}


def split_mask(index: pd.DatetimeIndex, name: str) -> pd.Series:
    a, b = SPLITS[name]
    return pd.Series((index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b)), index=index)


def split(obj, name: str):
    return obj[split_mask(obj.index, name).to_numpy()]
