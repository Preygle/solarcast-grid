import numpy as np
import pandas as pd
import pytest

from src.analysis import surplus as sp
from src.evaluation.metrics import regression_metrics
from src.features.build import build_features, check_no_leakage, default_spec, issue_time
from src.models.baselines import BASELINES


def _hourly(days=60, seed=1):
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(seed)
    h = pd.DataFrame({
        "demand_mw": 180_000 + 20_000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1000, len(idx)),
        "solar_mw": np.clip(40_000 * np.sin(np.pi * (idx.hour - 6) / 12), 0, None),
        "wind_mw": 8_000 + rng.normal(0, 500, len(idx)),
    }, index=idx)
    h["net_load_mw"] = h["demand_mw"] - h["solar_mw"] - h["wind_mw"]
    return h


def test_issue_times():
    t = pd.DatetimeIndex(["2024-03-02 00:00", "2024-03-02 23:00"])
    assert list(issue_time(t, "h1")) == list(t - pd.Timedelta(hours=1))
    assert (issue_time(t, "da") == pd.Timestamp("2024-03-01 09:00")).all()


@pytest.mark.parametrize("target", ["demand_mw", "solar_mw", "net_load_mw"])
@pytest.mark.parametrize("horizon", ["h1", "da"])
def test_no_future_leakage_by_perturbation(target, horizon):
    """Changing any value after the issue time must not change that row's features."""
    h = _hourly()
    spec = default_spec(target, horizon)
    check_no_leakage(spec)
    X, _, meta = build_features(h, spec)
    for t in X.index[24 * 20::97][:12]:
        tau = meta.loc[t, "issue_time"]
        h2 = h.copy()
        future = h2.index > tau
        h2.loc[future, ["demand_mw", "solar_mw", "wind_mw", "net_load_mw"]] *= 1.37
        X2, _, _ = build_features(h2, spec)
        pd.testing.assert_series_equal(X.loc[t], X2.loc[t], check_names=False)


def test_da_spec_rejects_short_lag():
    spec = default_spec("demand_mw", "da")
    spec.target_lags = [24]
    with pytest.raises(AssertionError):
        check_no_leakage(spec)


def test_baselines_only_use_past():
    h = _hourly()
    y = h["demand_mw"]
    for hz in ["h1", "da"]:
        for fn in BASELINES.values():
            p = fn(y, hz)
            y2 = y.copy(); y2.iloc[-24:] = 0
            p2 = fn(y2, hz)
            # forecasts for the last day are unchanged for h1 only up to the perturbed point
            cutoff = y.index[-24] if hz == "h1" else y.index[-24].normalize() + pd.Timedelta(days=1)
            assert np.allclose(p[p.index <= cutoff].dropna(), p2[p2.index <= cutoff].dropna())


def test_metrics_basic():
    y = pd.Series([100.0, 200.0, 300.0])
    m = regression_metrics(y, pd.Series([110.0, 190.0, 300.0], index=y.index), "demand_mw")
    assert m["mae"] == pytest.approx(20 / 3) and m["rmse"] == pytest.approx(np.sqrt(200 / 3))
    assert "mape_pct" in m
    assert "mape_pct" not in regression_metrics(y, y, "solar_mw")


def test_surplus_proxy_formula():
    idx = pd.date_range("2023-01-01", periods=24, freq="h")
    h = pd.DataFrame({"demand_mw": 100.0, "solar_mw": 0.0, "wind_mw": 10.0, "thermal_mw": 80.0,
                      "hydro_mw": 20.0, "nuclear_mw": 5.0}, index=idx)
    h.loc[idx[12], "solar_mw"] = 50.0
    h.loc[idx[0], "thermal_mw"] = 100.0  # daily max 100 -> floor = 0.5*100 + 5 + 20 = 75
    s = sp.surplus_proxy(h, theta=0.5, k=1.0)
    assert s.iloc[12] == pytest.approx(50 + 10 + 75 - 100)
    assert (s.drop(idx[12]) == 0).all()
