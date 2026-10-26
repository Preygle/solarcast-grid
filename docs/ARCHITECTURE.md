# Architecture

```
                         ┌─────────────────────── FORECASTING PIPELINE (real data) ───────────────────────┐
 Mendeley xlsx (raw) ──► src/data  ──► data/processed/hourly.parquet ──► src/features ──► src/training ──► models/*.joblib|pt
 Open-Meteo (weather) ─┘  (clean,        (OBSERVED/DERIVED, 1 h)          (lags, calendar,   (CV, tuning,      │
                           resample)                                        leak-safe)        early stop)     │
                                 │                                                                            ▼
                                 ├──► src/analysis (EDA, duck curve, surplus proxy) ──► results/eda/*      src/inference
                                 │                                                                     (load model, predict
                                 │                                                                      day-ahead profiles)
                                 │                                                                            │ FORECAST
                         └───────┼──────────────────────────────────────────────────────────────────────────┼────┘
                                 │ normalised observed profiles                                              │ normalised forecast profiles
                         ┌───────▼──────────────────────── GRID PIPELINE (simulation) ──────────────────────▼────┐
                         │ src/grid (33-bus pandapower net, NetworkX graph, centrality/paths/configs)            │
                         │      │                                                                                │
                         │ src/simulation (scenario sampler) ──► src/power_flow (AC PF, convergence, balance)    │
                         │      │                                    │ SIMULATED labels                          │
                         │      │                                    ▼                                           │
                         │      │                           src/receiving_power (loss / P_recv models)           │
                         │      ▼                                                                                │
                         │ src/optimization (24 h LP: battery, flex load, curtailment; LinDistFlow) ──► AC check  │
                         │      │ OPTIMISED + SIMULATED                                                          │
                         └──────┼────────────────────────────────────────────────────────────────────────────────┘
                                ▼
                       results/*.csv, results/figures  ──►  app/ (Streamlit + Plotly, reads artefacts only)
```

## Component contracts

| Component | Input | Output |
|---|---|---|
| `src/data` | raw xlsx | `data/processed/hourly.parquet` with columns `demand_mw, solar_mw, wind_mw, thermal_mw, hydro_mw, gas_mw, nuclear_mw, net_load_mw`, quality flags `*_imputed`, and `source_file`; `results/eda/preprocessing_log.json` |
| `src/features` | hourly frame, horizon spec | feature matrix X and aligned target y; `feature_spec.json` (names, lag set, issue time) |
| `src/training` / `src/models` | X/y splits | fitted model artefacts in `models/<task>/<model>.joblib` (LSTM: `.pt`) and `models/registry.json` |
| `src/evaluation` | predictions + truth | `results/model_comparison.csv`, `results/metrics.json`, `results/figures/*` |
| `src/inference` | registry + recent history | `forecast_day_ahead(date) → DataFrame[hour, solar_mw, demand_mw, wind_mw, net_load_mw, (p10, p90)]` |
| `src/grid` | — | pandapower net + `networkx.Graph`; topology metrics |
| `src/simulation` / `src/power_flow` | scenario parameters | `results/grid_scenarios/scenarios.parquet` (one row per converged PF, with scenario id, group, seed, config, inputs, outputs, convergence flag) |
| `src/receiving_power` | scenario table | loss / P_recv / loss-% models + `results/grid_model_comparison.csv` |
| `src/optimization` | day profiles (forecast and actual), battery/flex params | schedule DataFrame, solver status, objective terms; AC validation report; `results/optimization_comparison.csv` |
| `app/` | everything above (read-only) | interactive dashboard |

## Bridge between the pipelines

The two pipelines exchange **shapes, not magnitudes**. For a chosen day:
`feeder_load_t = P_feeder_peak × demand_t / max(demand over reference window)`,
`pv_t = P_pv_rated × solar_t / rolling-30-day max(solar)`.
The forecast versions use the same transforms on forecast values. This step is documented in `docs/OPTIMIZATION_METHODOLOGY.md` and is a scenario device: the feeder is not claimed to be any real Indian feeder.

## Technology choices
Python 3.13; pandas/numpy; scikit-learn, LightGBM, XGBoost, PyTorch (LSTM); pandapower + numba; NetworkX; PuLP + HiGHS; Streamlit + Plotly; pytest. The workspace was empty, so there was no existing framework to keep.
