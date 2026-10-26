"""Naive benchmark forecasts. Each uses only values at or before the issue time."""
from __future__ import annotations

import pandas as pd


def persistence(y: pd.Series, horizon: str) -> pd.Series:
    """h1: y(t-1). da: last value at issue time is a poor day-ahead guess, so the standard
    day-ahead persistence is the same hour of the most recent complete day (t-48h)."""
    return y.shift(1 if horizon == "h1" else 48).rename("persistence")


def seasonal_naive_daily(y: pd.Series, horizon: str) -> pd.Series:
    """Same hour on the most recent day available: t-24 (h1) or t-48 (da)."""
    return y.shift(24 if horizon == "h1" else 48).rename("seasonal_naive_24h")


def seasonal_naive_weekly(y: pd.Series, horizon: str) -> pd.Series:
    return y.shift(168).rename("seasonal_naive_168h")


def samehour_mean7(y: pd.Series, horizon: str) -> pd.Series:
    """Mean of the same clock-hour over the last 7 available days."""
    start = 1 if horizon == "h1" else 2
    return pd.concat([y.shift(24 * d) for d in range(start, start + 7)], axis=1).mean(axis=1).rename("samehour_mean_7d")


BASELINES = {
    "persistence": persistence,
    "seasonal_naive_24h": seasonal_naive_daily,
    "seasonal_naive_168h": seasonal_naive_weekly,
    "samehour_mean_7d": samehour_mean7,
}
