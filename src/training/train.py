"""Train, tune and compare forecasting models for one (target, horizon) task.

Protocol (identical for every model family):
1. Hyper-parameters and the target transform (none / level-ratio) are searched with models
   fitted on TRAIN and scored by MAE on VALIDATION. Boosting iterations come from early stopping
   on validation.
2. The best configuration of each family is refitted on TRAIN+VALIDATION (same iteration count)
   and evaluated once on TEST.
3. The task's selected model is the family with the lowest VALIDATION MAE (baselines included);
   test scores are never used for selection.
4. 80% prediction intervals: split-conformal, using absolute residuals of the train-only fit on
   validation, per hour of day (the 80th percentile of |residual|).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import mase_scale, regression_metrics
from src.features.build import build_features, check_no_leakage, default_spec
from src.features.splits import split_mask
from src.models.baselines import BASELINES
from src.models.lstm import LSTMForecaster
from src.models.tabular import LEVEL_COL, N_TRIALS, LevelRatioRegressor, build, sample_params

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "results" / "predictions"
FAMILIES = ["ridge", "random_forest", "hist_gbr", "lightgbm", "xgboost"]
BOOSTING = {"lightgbm", "xgboost"}
LSTM_CONFIGS = [{"hidden": 32}, {"hidden": 64}]
PI_LEVEL = 0.8


def postprocess(target: str, idx: pd.DatetimeIndex, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    if target == "solar_mw":
        p = np.clip(p, 0, None)
        p[np.isin(idx.hour, [20, 21, 22, 23, 0, 1, 2, 3])] = 0.0  # physically zero (see preprocessing)
    if target == "wind_mw":
        p = np.clip(p, 0, None)
    return p


def _fit_tabular(family, params, level_col, Xtr, ytr, Xva=None, yva=None, n_iter=None, target=""):
    p = dict(params)
    if family in BOOSTING and n_iter is not None:
        p["n_estimators"] = n_iter
    cols = [c for c in Xtr.columns if c != level_col]
    est = build(family, p, cols)
    if family == "xgboost" and Xva is not None:
        est.set_params(early_stopping_rounds=150, eval_metric="mae")
    model = LevelRatioRegressor(est, level_col, (f"{target}_lag", "tgt_") if level_col else ())
    kw = {}
    if Xva is not None and family == "lightgbm":
        kw = {"eval_set": [(Xva, yva)],
              "callbacks": [lgb.early_stopping(150, verbose=False)]}
    elif Xva is not None and family == "xgboost":
        kw = {"eval_set": [(Xva, yva)], "verbose": False}
    model.fit(Xtr, ytr, **kw)
    best_iter = None
    if Xva is not None and family == "lightgbm":
        best_iter = int(model.estimator_.best_iteration_ or model.estimator_.n_estimators)
    if Xva is not None and family == "xgboost":
        best_iter = int(model.estimator_.best_iteration + 1)
    return model, best_iter


def run_task(hourly: pd.DataFrame, target: str, horizon: str, weather: pd.DataFrame | None = None,
             families: list[str] | None = None, run_lstm: bool = True, tag: str = "",
             save: bool = True, log=print, spec=None) -> dict:
    families = families or FAMILIES
    spec = spec or default_spec(target, horizon)
    if weather is not None:
        spec.weather = list(weather.columns)
    check_no_leakage(spec)
    task = f"{target.replace('_mw', '')}_{horizon}{tag}"
    X, y, meta = build_features(hourly, spec, weather)
    m_tr, m_va, m_te = (split_mask(X.index, s).to_numpy() for s in ["train", "val", "test"])
    m_trva = m_tr | m_va
    Xtr, ytr, Xva, yva, Xte, yte = X[m_tr], y[m_tr], X[m_va], y[m_va], X[m_te], y[m_te]
    scale = mase_scale(hourly.loc[hourly.index <= Xtr.index.max(), target])

    rows, preds_val, preds_test, fitted, trials = [], {}, {}, {}, []

    def record(name, pv, pt, cfg):
        preds_val[name], preds_test[name] = pv, pt
        for part, yy, pp in [("val", yva, pv), ("test", yte, pt)]:
            rows.append({"task": task, "target": target, "horizon": horizon, "model": name, "split": part,
                         **regression_metrics(yy, pd.Series(pp, index=yy.index), target, scale), "config": json.dumps(cfg)})

    for name, fn in BASELINES.items():
        if horizon == "da" and name == "persistence":
            continue  # identical to seasonal_naive_24h for day-ahead (both t-48)
        b = fn(hourly[target], horizon)
        record(name, b.reindex(yva.index).to_numpy(), b.reindex(yte.index).to_numpy(), {"baseline": name})

    for fam in families:
        t0 = time.time()
        best = None
        for level_col in [None] + ([LEVEL_COL[target]] if LEVEL_COL[target] in X.columns else []):
            for params in sample_params(fam, N_TRIALS[fam]):
                model, it = _fit_tabular(fam, params, level_col, Xtr, ytr, Xva, yva, target=target)
                pv = postprocess(target, Xva.index, model.predict(Xva))
                mae = float(np.nanmean(np.abs(pv - yva.to_numpy())))
                trials.append({"task": task, "family": fam, "level_col": level_col, "params": params,
                               "n_iter": it, "val_mae": mae})
                if best is None or mae < best[0]:
                    best = (mae, params, level_col, it, model, pv)
        mae, params, level_col, it, model_tr, pv = best
        final, _ = _fit_tabular(fam, params, level_col, X[m_trva], y[m_trva], n_iter=it, target=target)
        pt = postprocess(target, Xte.index, final.predict(Xte))
        cfg = {"params": params, "level_col": level_col, "n_iter": it}
        record(fam, pv, pt, cfg)
        fitted[fam] = {"model": final, "model_train_only": model_tr, "config": cfg}
        log(f"  {task} {fam}: val MAE {mae:,.0f} | test MAE {np.nanmean(np.abs(pt - yte.to_numpy())):,.0f} ({time.time() - t0:.0f}s)")

    if run_lstm:
        t0 = time.time()
        lv_col = LEVEL_COL[target]
        lv = lambda Z: np.where(np.isfinite(Z[lv_col]) & (Z[lv_col] > 0), Z[lv_col], np.nanmedian(Z[lv_col]))
        best = None
        for cfg in LSTM_CONFIGS:
            net = LSTMForecaster(**cfg)
            net.fit_with_validation(hourly, Xtr.index, meta.loc[Xtr.index, "issue_time"], lv(Xtr), ytr.to_numpy(),
                                    Xva.index, meta.loc[Xva.index, "issue_time"], lv(Xva), yva.to_numpy())
            pv = postprocess(target, Xva.index, net.predict(hourly, Xva.index, meta.loc[Xva.index, "issue_time"], lv(Xva)))
            mae = float(np.nanmean(np.abs(pv - yva.to_numpy())))
            trials.append({"task": task, "family": "lstm", "params": cfg, "n_iter": net.best_epochs_, "val_mae": mae})
            if best is None or mae < best[0]:
                best = (mae, cfg, net, pv)
        mae, cfg, net_tr, pv = best
        Xtv = X[m_trva]
        final = LSTMForecaster(**cfg).refit(hourly, Xtv.index, meta.loc[Xtv.index, "issue_time"], lv(Xtv),
                                            y[m_trva].to_numpy(), epochs=net_tr.best_epochs_)
        pt = postprocess(target, Xte.index, final.predict(hourly, Xte.index, meta.loc[Xte.index, "issue_time"], lv(Xte)))
        c = {"params": cfg, "level_col": lv_col, "epochs": net_tr.best_epochs_}
        record("lstm", pv, pt, c)
        fitted["lstm"] = {"model": final, "model_train_only": net_tr, "config": c}
        log(f"  {task} lstm: val MAE {mae:,.0f} | test MAE {np.nanmean(np.abs(pt - yte.to_numpy())):,.0f} ({time.time() - t0:.0f}s)")

    comp = pd.DataFrame(rows)
    val_rank = comp[comp.split == "val"].sort_values("mae")
    best_name = val_rank.iloc[0]["model"]

    # Conformal half-widths per hour from the selected model's out-of-sample validation residuals.
    res = (pd.Series(preds_val[best_name], index=yva.index) - yva).abs()
    q = res.groupby(res.index.hour).quantile(PI_LEVEL)
    half = q.reindex(yte.index.hour).to_numpy()
    pt = np.asarray(preds_test[best_name])
    lo = pt - half
    if target in ("solar_mw", "wind_mw"):
        lo = np.clip(lo, 0, None)
    cover = float(np.nanmean((yte.to_numpy() >= lo) & (yte.to_numpy() <= pt + half)))

    pred_df = pd.DataFrame({"y": pd.concat([yva, yte]),
                            "split": ["val"] * len(yva) + ["test"] * len(yte)})
    for name in preds_val:
        pred_df[name] = np.concatenate([preds_val[name], preds_test[name]])
    pred_df["pi_lo"] = np.nan
    pred_df["pi_hi"] = np.nan
    pred_df.loc[yte.index, "pi_lo"] = lo
    pred_df.loc[yte.index, "pi_hi"] = pt + half

    summary = {"task": task, "target": target, "horizon": horizon, "selected_model": best_name,
               "selected_by": "lowest validation MAE", "pi_level": PI_LEVEL,
               "pi_empirical_coverage_test": cover, "pi_halfwidth_by_hour": {int(k): float(v) for k, v in q.items()},
               "feature_spec": spec.to_dict(), "features": list(X.columns),
               "n_train": int(m_tr.sum()), "n_val": int(m_va.sum()), "n_test": int(m_te.sum()),
               "mase_scale": scale}
    if save:
        tdir = MODELS_DIR / task
        tdir.mkdir(parents=True, exist_ok=True)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        for fam, d in fitted.items():
            joblib.dump({"model": d["model"], "config": d["config"], "features": list(X.columns),
                         "spec": spec.to_dict(), "target": target, "horizon": horizon}, tdir / f"{fam}.joblib")
        pred_df.to_parquet(PRED_DIR / f"{task}.parquet")
        (tdir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(trials).to_csv(tdir / "tuning_trials.csv", index=False)
    return {"comparison": comp, "summary": summary, "predictions": pred_df, "fitted": fitted,
            "X": X, "y": y, "meta": meta}
