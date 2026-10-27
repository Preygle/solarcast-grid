import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.models.lstm import LSTMForecaster
from src.models.tabular import LevelRatioRegressor

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "models" / "registry.json"


def test_level_ratio_regressor_scale_free():
    rng = np.random.default_rng(0)
    lvl = rng.uniform(10, 100, 500)
    X = pd.DataFrame({"tgt_level": lvl, "solar_mw_lag48": 0.5 * lvl, "hour": rng.integers(0, 24, 500)})
    y = 0.8 * lvl
    m = LevelRatioRegressor(LinearRegression(), "tgt_level", ("solar_mw_lag", "tgt_")).fit(X, y)
    X2 = X * [3, 3, 1]  # triple the level: prediction must triple (extrapolation beyond training)
    assert np.allclose(m.predict(X2), 3 * y, rtol=1e-6)
    assert "tgt_level" not in m._transform(X).columns


def test_lstm_pickle_roundtrip():
    idx = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    h = pd.DataFrame({"demand_mw": 1e5 + 1e4 * np.sin(np.arange(len(idx)) / 4),
                      "solar_mw": np.clip(np.sin(np.arange(len(idx)) / 4), 0, None) * 3e4,
                      "wind_mw": 5e3}, index=idx)
    rows = idx[200:300]
    issue = rows - pd.Timedelta(hours=1)
    lv = np.full(len(rows), 1e5)
    m = LSTMForecaster(hidden=8, max_epochs=1).refit(h, rows, issue, lv, h.loc[rows, "demand_mw"].to_numpy(), epochs=1)
    p1 = m.predict(h, rows, issue, lv)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.allclose(p1, m2.predict(h, rows, issue, lv))


@pytest.mark.skipif(not REG.exists(), reason="models not trained")
def test_saved_models_reload_and_predict_unseen_day():
    from src.data.preprocess import load_hourly
    from src.inference.forecast import Forecaster
    h = load_hourly()
    fc = Forecaster()
    day = "2025-05-20"
    out = fc.day_ahead(h, day)
    assert len(out) == 24 and out[["solar_mw", "demand_mw", "wind_mw"]].notna().all().all()
    assert (out.solar_mw >= 0).all() and (out.solar_mw_pi_lo <= out.solar_mw).all()
    assert (out.issue_time == pd.Timestamp("2025-05-19 09:00")).all()
    # forecasts must not change if data after the issue time is altered (no leakage at inference)
    h2 = h.copy()
    h2.loc[h2.index > pd.Timestamp("2025-05-19 09:00"), ["demand_mw", "solar_mw", "wind_mw", "net_load_mw"]] *= 1.5
    out2 = fc.day_ahead(h2, day)
    assert np.allclose(out.demand_mw, out2.demand_mw) and np.allclose(out.solar_mw, out2.solar_mw)
    # units: all-India demand forecast in MW
    assert 1e5 < out.demand_mw.mean() < 3e5


@pytest.mark.skipif(not REG.exists(), reason="models not trained")
def test_registry_consistency():
    reg = json.loads(REG.read_text())
    for task, e in reg.items():
        assert e["selected_model"]
        if e["artifact"]:
            assert (ROOT / e["artifact"]).exists(), task


def test_forecaster_missing_registry(tmp_path):
    from src.inference.forecast import Forecaster, ModelUnavailable
    with pytest.raises(ModelUnavailable):
        Forecaster(tmp_path / "nope.json")
