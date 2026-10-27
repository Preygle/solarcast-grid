"""Predict technical losses, loss % and receiving-end power from operating conditions.

Labels are SIMULATED (pandapower AC PF). Inputs are operating conditions only: load totals and
zonal loads (P), reactive load total, PV injection per site, battery power and the radial
configuration (one-hot + graph features). Power-flow outputs (line flows, voltages, slack power)
are never used as inputs.

Evaluation: GroupShuffleSplit on scenario family, so test families (day x configuration x PV
capacities x battery profile) are never seen in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.grid.network import PV_BUSES
from src.simulation.scenarios import ZONES

TARGETS = {"p_loss_mw": "Technical losses [MW]", "p_recv_mw": "Receiving-end power at bus 0 [MW]",
           "loss_pct": "Loss % of sent power"}
FORBIDDEN = {"p_loss_mw", "p_recv_mw", "loss_pct", "p_import_mw", "p_export_mw", "p_sent_mw",
             "p_delivered_mw", "q_ext_mvar", "vm_min_pu", "vm_max_pu", "line_loading_max_pct",
             "balance_residual_mw", "p_load_mw", "p_pv_mw", "p_batt_mw"}


def feature_frame(df: pd.DataFrame, n_configs: int, config_feats: pd.DataFrame | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["load_p_total_mw"] = df["load_p_total_mw"]
    X["load_q_total_mvar"] = df["load_q_total_mvar"]
    for z in ZONES:
        X[f"load_{z}_mw"] = df[f"load_{z}_mw"]
    pv_cols = [f"pv_{b}_mw" for b in PV_BUSES]
    for c in pv_cols:
        X[c] = df[c]
    X["pv_total_mw"] = df[pv_cols].sum(axis=1)
    X["battery_mw"] = df["battery_mw"]
    # Net injection known before running PF: sum of sources minus loads (no losses).
    X["net_injection_mw"] = X["pv_total_mw"] + X["battery_mw"] - X["load_p_total_mw"]
    for k in range(n_configs):
        X[f"cfg_{k}"] = (df["config_id"] == k).astype(int)
    if config_feats is not None:
        X = X.join(df[["config_id"]].merge(config_feats, left_on="config_id", right_index=True, how="left")
                   .drop(columns="config_id").set_index(df.index))
    assert not FORBIDDEN & set(X.columns), "target leakage"
    return X


def models() -> dict:
    return {
        # Physics-motivated baseline: I^2R losses are quadratic in flows, so a ridge regression on
        # second-order polynomial features of the injections is the interpretable reference.
        "quadratic_ridge": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                         RidgeCV(alphas=np.logspace(-4, 2, 13))),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5,
                                               n_jobs=-1, random_state=42),
        "hist_gbr": HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
                                                  random_state=42),
        "lightgbm": lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=31,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                      random_state=42, verbose=-1),
    }


def group_split(df: pd.DataFrame, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, rest = next(gss.split(df, groups=df["family"]))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    va, te = next(gss2.split(rest_df, groups=rest_df["family"]))
    return df.index[tr], rest_df.index[va], rest_df.index[te]


def metrics(y, p) -> dict:
    e = np.asarray(p) - np.asarray(y)
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((np.asarray(y) - np.mean(y)) ** 2)),
            "max_abs_err": float(np.max(np.abs(e)))}
