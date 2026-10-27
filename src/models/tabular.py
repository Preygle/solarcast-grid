"""Tabular regressors and their bounded hyper-parameter spaces.

``LevelRatioRegressor`` addresses non-stationarity: tree ensembles cannot extrapolate beyond the
training range, but all-India solar capacity grows every month (test-period peaks exceed the
training maximum). The wrapped model learns y / level, where ``level`` is a feature known at
the issue time (e.g. the rolling 30-day max of solar), and predictions are multiplied back.
Whether to use it is a hyper-parameter chosen on the validation set.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

SEED = 42
LEVEL_COL = {"solar_mw": "tgt_max720_at_issue", "demand_mw": "tgt_mean168_at_issue",
             "net_load_mw": "tgt_mean168_at_issue", "wind_mw": "tgt_max720_at_issue"}


class LevelRatioRegressor(BaseEstimator, RegressorMixin):
    """With ``level_col`` set, the model is scale-free: the target AND the target's own lag/rolling
    features (``scale_prefixes``) are divided by the level; the level column itself is dropped."""

    def __init__(self, estimator=None, level_col: str | None = None, scale_prefixes: tuple = ()):
        self.estimator = estimator
        self.level_col = level_col
        self.scale_prefixes = scale_prefixes

    def _level(self, X: pd.DataFrame) -> np.ndarray:
        lv = X[self.level_col].to_numpy(dtype=float)
        return np.where(np.isfinite(lv) & (lv > 0), lv, np.nanmedian(lv))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.level_col is None:
            return X
        lv = self._level(X)
        Z = X.drop(columns=[self.level_col]).copy()
        for c in Z.columns:
            if c.startswith(tuple(self.scale_prefixes)):
                Z[c] = Z[c].to_numpy(dtype=float) / lv
        return Z

    def fit(self, X, y, **fit_kw):
        self.estimator_ = clone(self.estimator)
        if self.level_col is None:
            self.estimator_.fit(X, y, **fit_kw)
        else:
            if "eval_set" in fit_kw:  # early-stopping data must be on the same (ratio) scale
                fit_kw["eval_set"] = [(self._transform(Xv), np.asarray(yv) / self._level(Xv))
                                      for Xv, yv in fit_kw["eval_set"]]
            self.estimator_.fit(self._transform(X), np.asarray(y) / self._level(X), **fit_kw)
        return self

    def predict(self, X):
        p = self.estimator_.predict(self._transform(X))
        return p if self.level_col is None else p * self._level(X)


def ridge_model(alpha: float = 1.0):
    # One-hot hour and day-of-week so the linear model can represent the diurnal/weekly shape.
    cat = ["hour", "dow"]
    def make(columns):
        num = [c for c in columns if c not in cat]
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
        return make_pipeline(pre, Ridge(alpha=alpha))
    return make


def build(family: str, params: dict, columns: list[str]):
    p = dict(params)
    if family == "ridge":
        return ridge_model(p.get("alpha", 1.0))(columns)
    if family == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=p.get("n_estimators", 300), max_depth=p.get("max_depth"),
            min_samples_leaf=p.get("min_samples_leaf", 2), max_features=p.get("max_features", 0.5),
            n_jobs=-1, random_state=SEED))
    if family == "hist_gbr":
        return HistGradientBoostingRegressor(
            learning_rate=p.get("learning_rate", 0.05), max_iter=p.get("max_iter", 2000),
            max_leaf_nodes=p.get("max_leaf_nodes", 31), min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0), early_stopping=False, random_state=SEED)
    if family == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 3000), learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 31), min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.8), subsample_freq=1, colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 0.0), objective=p.get("objective", "l1"), metric="l1",
            random_state=SEED, n_jobs=-1, verbose=-1)
    if family == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 3000), learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 6), min_child_weight=p.get("min_child_weight", 1),
            subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0), objective=p.get("objective", "reg:absoluteerror"),
            tree_method="hist", random_state=SEED, n_jobs=-1)
    raise ValueError(family)


# Bounded search spaces (sampled with a fixed seed). Iteration counts for boosting are set by
# early stopping on the validation set, then reused when refitting on train+val.
SPACES = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 3, 10],
                      "max_features": [0.33, 0.6]},
    "hist_gbr": {"learning_rate": [0.03, 0.06], "max_leaf_nodes": [15, 31, 63],
                 "min_samples_leaf": [20, 50], "l2_regularization": [0.0, 1.0],
                 "max_iter": [300, 600, 1200]},
    "lightgbm": {"learning_rate": [0.02, 0.05], "num_leaves": [15, 31, 63], "min_child_samples": [10, 30, 60],
                 "colsample_bytree": [0.6, 0.9], "reg_lambda": [0.0, 1.0], "objective": ["l1", "l2"]},
    "xgboost": {"learning_rate": [0.02, 0.05], "max_depth": [4, 6, 8], "min_child_weight": [1, 5],
                "colsample_bytree": [0.6, 0.9], "objective": ["reg:absoluteerror", "reg:squarederror"]},
}
N_TRIALS = {"ridge": 4, "random_forest": 3, "hist_gbr": 6, "lightgbm": 10, "xgboost": 6}


def sample_params(family: str, n: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    space = SPACES[family]
    keys = list(space)
    combos, seen = [], set()
    total = int(np.prod([len(space[k]) for k in keys]))
    while len(combos) < min(n, total):
        c = tuple(int(rng.integers(len(space[k]))) for k in keys)
        if c not in seen:
            seen.add(c)
            combos.append({k: space[k][i] for k, i in zip(keys, c)})
    return combos
