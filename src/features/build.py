"""Leak-safe feature construction for hourly forecasting.

Every row is a *target hour* t (hourly mean over [t, t+1h), IST). Each row has an
*issue time* tau: the last hourly timestamp whose value is known when the forecast is made.

* ``h1``  (1-hour-ahead): forecast made at t for hour t, so tau = t - 1h.
* ``da``  (day-ahead): forecast made at 10:00 IST on day D for all hours of D+1, so
  tau = D 09:00 (the 09:00-10:00 hour has just completed). Leads are 15..38 h from tau.

All features are computed from values at timestamps <= tau. ``check_no_leakage`` verifies this
for every lag/rolling feature by construction (minimum offset >= lead).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SERIES = ["demand_mw", "solar_mw", "wind_mw", "net_load_mw"]
DA_ISSUE_HOUR = 9  # last complete hour at issue time (10:00 IST)


@dataclass
class FeatureSpec:
    target: str
    horizon: str  # "h1" | "da"
    target_lags: list[int] = field(default_factory=list)
    exog_lags: dict[str, list[int]] = field(default_factory=dict)
    issue_features: bool = True
    calendar: bool = True
    weather: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def default_spec(target: str, horizon: str) -> FeatureSpec:
    others = [s for s in ["demand_mw", "solar_mw", "wind_mw"] if s != target]
    if horizon == "h1":
        return FeatureSpec(target, horizon,
                           target_lags=[1, 2, 3, 24, 48, 168],
                           exog_lags={o: [1, 24] for o in others})
    return FeatureSpec(target, horizon,
                       target_lags=[48, 72, 168],
                       exog_lags={o: [48, 168] for o in others})


def issue_time(index: pd.DatetimeIndex, horizon: str) -> pd.DatetimeIndex:
    if horizon == "h1":
        return index - pd.Timedelta(hours=1)
    if horizon == "da":
        return index.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=DA_ISSUE_HOUR)
    raise ValueError(horizon)


def min_lead_hours(horizon: str) -> int:
    """Smallest lag L such that y(t-L) is known at the issue time for *every* target hour:
    1 for h1; for da the largest lead t - tau is 23:00 of D+1 minus 09:00 of D = 38 h."""
    return 1 if horizon == "h1" else 24 - DA_ISSUE_HOUR + 23


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    return pd.DataFrame({
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": idx.dayofweek.to_numpy(),
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "month": idx.month.to_numpy(),
    }, index=idx)


def _at(series: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(when).to_numpy()


def build_features(hourly: pd.DataFrame, spec: FeatureSpec,
                   weather: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, meta) aligned on target timestamps. Rows with NaN target are dropped;
    NaN features are left for the model (tree models handle them; others impute)."""
    idx = hourly.index
    tau = issue_time(idx, spec.horizon)
    lead = min_lead_hours(spec.horizon)
    X = pd.DataFrame(index=idx)
    y = hourly[spec.target]

    for L in spec.target_lags:
        if spec.horizon == "da" and L < lead:
            raise ValueError(f"lag {L} < minimum lead {lead} for day-ahead")
        X[f"{spec.target}_lag{L}"] = y.shift(L)
    for col, lags in spec.exog_lags.items():
        for L in lags:
            X[f"{col}_lag{L}"] = hourly[col].shift(L)

    if spec.issue_features:
        # Statistics of the target over windows ending at the issue time.
        roll24 = y.rolling(24, min_periods=18)
        roll168 = y.rolling(168, min_periods=120)
        X["tgt_last_at_issue"] = _at(y, tau)
        X["tgt_mean24_at_issue"] = _at(roll24.mean(), tau)
        X["tgt_std24_at_issue"] = _at(roll24.std(), tau)
        X["tgt_max24_at_issue"] = _at(roll24.max(), tau)
        X["tgt_mean168_at_issue"] = _at(roll168.mean(), tau)
        # Level / capacity proxy: rolling 30-day max of the target (captures solar build-out).
        X["tgt_max720_at_issue"] = _at(y.rolling(720, min_periods=480).max(), tau)
        # Same clock-hour mean over the last 7 available days (seasonal profile at issue time).
        same_hour = pd.concat([y.shift(24 * d) for d in range(1, 8)], axis=1).mean(axis=1)
        shift_h = 0 if spec.horizon == "h1" else 24  # da: D+1 hour h -> use days ending D-1
        X["tgt_samehour_mean7d"] = same_hour.shift(shift_h)

    if spec.calendar:
        X = X.join(calendar_features(idx))
    if spec.weather and weather is not None:
        X = X.join(weather[spec.weather])

    meta = pd.DataFrame({"issue_time": tau, "horizon": spec.horizon}, index=idx)
    keep = y.notna()
    return X[keep], y[keep], meta[keep]


def lag_offsets(spec: FeatureSpec) -> dict[str, int]:
    """Minimum look-back (hours before target) of each lag-type feature, for leakage tests."""
    out = {f"{spec.target}_lag{L}": L for L in spec.target_lags}
    for col, lags in spec.exog_lags.items():
        out.update({f"{col}_lag{L}": L for L in lags})
    return out


def check_no_leakage(spec: FeatureSpec) -> None:
    lead = min_lead_hours(spec.horizon)
    bad = {k: v for k, v in lag_offsets(spec).items() if v < lead}
    if bad:
        raise AssertionError(f"features look past the issue time: {bad}")
