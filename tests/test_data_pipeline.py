import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as pp
from src.features.splits import SPLITS, split_mask


def _raw(n=600, step="5min"):
    idx = pd.date_range("2022-04-01", periods=n, freq=step)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "demand_mw": 180_000 + rng.normal(0, 300, n),
        "solar_raw_mw": np.clip(30_000 * np.sin(np.linspace(0, np.pi, n)), -50, None),
        "wind_mw": 8_000 + rng.normal(0, 50, n),
        "thermal_mw": 130_000 + rng.normal(0, 200, n),
        "hydro_mw": 18_000 + rng.normal(0, 50, n),
        "gas_mw": 3_000 + rng.normal(0, 20, n),
        "nuclear_mw": 5_000 + rng.normal(0, 10, n),
    }, index=idx)
    return df


def test_dropout_zero_rows_masked():
    df = _raw()
    df.iloc[100, :] = 0.0
    out, log = pp.clean_frame(df)
    assert log["dropout_rows_demand_zero"] == 1
    assert out.iloc[100][["demand_mw", "thermal_mw", "solar_mw"]].isna().all()


def test_spike_and_bounds_masked():
    df = _raw()
    df.iloc[200, 0] = 466_673.0   # observed impossible demand value
    df.iloc[300, 0] = 150_000.0   # in-bounds spike (-30 GW)
    out, log = pp.clean_frame(df)
    assert np.isnan(out["demand_mw"].iloc[200]) and np.isnan(out["demand_mw"].iloc[300])
    assert log["demand_mw_out_of_bounds"] == 1
    assert out["demand_mw"].notna().sum() >= len(df) - 3  # nothing else lost


def test_negative_solar_clipped():
    df = _raw()
    df["solar_raw_mw"] = -40.0
    out, log = pp.clean_frame(df)
    assert (out["solar_mw"] == 0).all() and log["solar_negative_clipped"] == len(df)


def test_hourly_coverage_rule():
    df = _raw(n=24)
    df.iloc[:7, df.columns.get_loc("demand_mw")] = np.nan  # 7/12 samples missing in first hour
    out, _ = pp.clean_frame(df)
    hourly = pp.to_hourly(out)
    assert np.isnan(hourly["demand_mw"].iloc[0])
    assert not np.isnan(hourly["demand_mw"].iloc[1])
    # hourly mean equals the mean of the samples in [t, t+1h)
    assert hourly["wind_mw"].iloc[1] == pytest.approx(out["wind_mw"].iloc[12:24].mean())


def test_flatline_mask():
    s = pd.Series([1.0, 2.0, 5.0, 5.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    m = pp.flatline_mask(s)
    assert m.tolist() == [False, False, True, True, True, False, False, False, False, False]


def test_seasonal_hampel_flags_stuck_night_value():
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    s = pd.Series(np.where((idx.hour >= 7) & (idx.hour <= 17), 40_000.0, 0.0), index=idx)
    s.iloc[24 * 15 + 22] = 4_800.0   # stuck value at 22:00
    f = pp.seasonal_hampel(s)
    assert f.iloc[24 * 15 + 22] and f.sum() == 1


def test_splits_chronological_and_disjoint():
    idx = pd.date_range("2021-09-01", "2025-06-24 23:00", freq="h")
    masks = {k: split_mask(idx, k) for k in SPLITS}
    assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()
    assert idx[masks["train"]].max() < idx[masks["val"]].min() < idx[masks["test"]].min()


@pytest.mark.skipif(not pp.HOURLY_PATH.exists(), reason="processed data not built")
def test_processed_table_integrity():
    h = pp.load_hourly()
    assert h.index.is_monotonic_increasing and not h.index.duplicated().any()
    assert (h.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    nl = h["demand_mw"] - h["solar_mw"] - h["wind_mw"]
    assert np.allclose(nl.dropna(), h["net_load_mw"].dropna())
    assert (h["solar_mw"].dropna() >= 0).all()
    night = h.index.hour.isin(pp.NIGHT_HOURS)
    assert (h.loc[night, "solar_mw"].dropna() == 0).all()
    for c, (lo, hi) in pp.BOUNDS.items():
        col = "solar_mw" if c == "solar_raw_mw" else c
        v = h[col].dropna()
        assert v.min() >= min(lo, 0) and v.max() <= hi, c
    # Units sanity: all-India demand is O(1e5) MW, i.e. values are MW not GW or kW.
    assert 1e5 < h["demand_mw"].median() < 3e5
    assert h["demand_mw"].isna().mean() < 0.01
