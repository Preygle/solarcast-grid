# SolarCast-Grid — Project Specification

## 1. Research question

> Can forecasts of all-India solar generation and electricity demand, built from real Grid-India SCADA data, be accurate enough to inform a constrained storage and flexible-load schedule? If so, does that schedule measurably reduce solar curtailment and technical losses in a physically simulated distribution feeder, compared with operation without the schedule?

Predictive analytics is the primary contribution. Power flow, graph analysis and optimisation are used to test whether the forecasts are useful.

## 2. Objectives

1. Build a reproducible pipeline for the Grid-India SCADA data (Sep 2021 – Jun 2025): cleaning, documented resampling to hourly.
2. Characterise the Indian duck curve with real data: net load, evening ramps, seasonal shape.
3. Forecast solar, demand and net load at 1 h and day-ahead horizons. Compare baselines, linear, tree ensembles, gradient boosting and an LSTM under one protocol.
4. Define and estimate a scenario-based surplus-solar-risk proxy.
5. Model a benchmark distribution feeder as a graph, analyse its topology, and generate labelled AC power-flow scenarios.
6. Predict technical losses, loss percentage and receiving-end power on held-out scenario groups.
7. Formulate a forecast-informed LP (battery + flexible load + curtailment) with linearised network constraints. Validate every schedule with full AC power flow and compare it with a baseline.

## 3. Data provenance classes

Every table, chart and metric is tagged with one of:

| Tag | Meaning |
|---|---|
| **OBSERVED** | Grid-India SCADA values (after documented cleaning) |
| **DERIVED** | Arithmetic on observed values (net load, ramps, hourly means) |
| **FORECAST** | Model output |
| **SIMULATED** | pandapower AC power-flow results on the 33-bus benchmark |
| **OPTIMISED** | LP decisions under stated assumptions |

## 4. Prediction targets

Common conventions: all power in MW, hourly mean values, timestamp = start of hour in IST. Energy = MWh (hourly mean MW × 1 h).

Splits for the Phase 2/3 forecasting work are chronological. **Train** 2021-09-01 → 2024-06-30, **validation** 2024-07-01 → 2024-12-31, **test** 2025-01-01 → 2025-06-24. The test period is used only once, for final reporting.

### 4.1 Solar generation forecast
- Target: `solar_mw(t+h)`, all-India, clipped at ≥ 0 MW.
- Horizons: h = 1 h (issued every hour) and day-ahead. For day-ahead, the forecast is issued at 10:00 IST on day D for all 24 hours of D+1, so leads are 14–37 h. Only data up to 10:00 on day D is usable.
- Inputs: calendar (hour, day-of-year harmonics, weekday), lags available at issue time, rolling statistics of past values, and a capacity-growth proxy (rolling 30-day max of solar). Weather appears only in the separate weather experiment.
- Metrics: MAE, RMSE (MW), R², nMAE = MAE / mean(actual) over daylight hours, MASE vs seasonal-naive. MAPE is not used because solar is zero at night (Hyndman & Koehler 2006).

### 4.2 Demand forecast
- Target: `demand_mw(t+h)` (demand met, NLDC). Same horizons and evaluation design. Metrics: MAE, RMSE, R², MAPE (demand is far from 0, so MAPE is valid here), MASE.

### 4.3 Net-load forecast
- Definition: `net_load = demand − solar − wind` (variable renewables only; hydro is treated as dispatchable).
- Two routes are compared: (a) direct model, (b) demand forecast − solar forecast − wind forecast (wind via its own model). Metrics as for demand, plus error at the daily minimum and on the evening ramp.

### 4.4 Surplus-solar risk (proxy — not actual curtailment)
The dataset has no unit-commitment, reserve, inter-regional constraint or curtailment records, so actual surplus cannot be observed. We define:

- Inflexible floor `F_t = θ · Thermal_committed,d + Nuclear_t + Hydro_min,d`. Here `Thermal_committed,d` = max thermal output on day d (a proxy for capacity online), θ ∈ {0.55, 0.40} is the CEA technical-minimum share (R2), and `Hydro_min,d` is the minimum hourly hydro on day d (proxy for run-of-river/must-run hydro).
- Surplus proxy under solar scale-up factor k: `S_t(k) = max(0, k·Solar_t + Wind_t + F_t − Demand_t)` in MW. Gas and the thermal output above the floor are treated as fully flexible (can be backed down to zero / to the floor).
- k = 1 checks the observed system; k ∈ {1.5, 2, 3} represents future solar build-out with today's demand and fleet.
- Outputs: hours with S>0, MWh/day, and a **risk label** `1[S_t(k) > 0]` for classification forecasting.
- Valid only for 2021-09 → 2023-12 (generation-mix columns exist only there). Stated as an assumption-driven scenario indicator, not an observation.

### 4.5 Technical-loss prediction (SIMULATED labels)
- Network: Baran & Wu 33-bus feeder (12.66 kV), with PV plants at 4 buses, a battery at 1 bus, and selectable tie-switch configurations.
- Label: `P_loss = Σ_lines (p_from + p_to)` from converged pandapower AC power flow [MW]. Loss % = `P_loss / P_sent × 100`, where `P_sent = P_import_bus0⁺ + Σ P_pv + P_batt_discharge⁺` (all power injected into the feeder).
- Inputs: total and per-zone load (P, Q), PV injection per site, battery power, configuration id, plus graph-derived features of the configuration. **Not allowed as inputs:** line flows, voltages, or the slack result (these leak the target).

### 4.6 Receiving-end power prediction (SIMULATED labels)
- Receiving-end bus: **bus 0**, the 12.66 kV grid supply point where the feeder meets the upstream utility (the offtaker). Positive `P_recv` = power the upstream utility receives from the feeder (export during solar surplus). Negative = import.
- `P_recv = −P_ext_grid` from AC power flow. Identity checked on every scenario: `Σ P_pv + P_batt − Σ P_load − P_loss − P_recv ≈ 0` (tolerance 1e-6 MW).
- Also reported: `P_delivered = Σ P_load + max(P_recv, 0)` = power that reached a sink (loads plus upstream utility), so `P_delivered = P_sent − P_loss` holds.
- Evaluation: grouped split by scenario family. Whole days/configurations are held out, so test rows never share a scenario seed with training rows. Metrics: MAE, RMSE, R², plus checks against fresh power-flow runs on new random conditions. Prediction intervals come from split-conformal or quantile gradient boosting.

## 5. Methodology summary

| Stage | Method | Key alternatives considered | Why |
|---|---|---|---|
| Resampling | Hourly mean of 5-min/10-s data after outlier masking | Snapshot at :00 | Mean is robust to 10-s noise; energy-consistent (MW·h = MWh) |
| Outlier handling | Zeros → NaN where impossible; Hampel filter (rolling median ± 6·MAD) on raw resolution; gaps ≤ 2 h interpolated in time and flagged; longer gaps left NaN | Drop rows | Keeps data, every change is recorded |
| Forecasting | Persistence, seasonal-naive, Ridge, Random Forest, HistGradientBoosting, LightGBM, XGBoost, LSTM | ARIMA/Prophet | Tree GBMs are strong on tabular lagged features; LSTM is the sequence benchmark; all share one protocol |
| Tuning | Bounded random/grid search on validation; early stopping | Test-set tuning (forbidden) | Prevents leakage |
| Grid | pandapower AC Newton-Raphson | DC power flow | DC ignores losses, which are the target here |
| Graph | NetworkX: impedance-weighted shortest paths, betweenness, bridges, spanning-tree (radial) configurations | Dijkstra-as-optimiser | Graph metrics identify candidates; AC power flow evaluates them |
| Optimisation | 24 h LP (PuLP + HiGHS): battery SoC, charge/discharge, flexible-load shifting, PV curtailment, export limit, LinDistFlow voltage & flow constraints; ex-post AC validation | MILP / AC-OPF | LP is tractable and auditable; AC check exposes linearisation error honestly |

## 6. Planned experiments

| ID | Experiment | Compared |
|---|---|---|
| E1 | Duck-curve characterisation | Seasonal net-load profiles, evening ramp (max 3-h increase 16:00–21:00), daily min/max; 2021→2025 trend |
| E2 | Baselines | Persistence, seasonal-naive (24 h, 168 h) |
| E3 | Model comparison (solar, demand, net load, wind) × (1 h, day-ahead) | All Phase 3 models, same features and splits |
| E4 | Feature ablation | No lags / no rolling / + weather (labelled) |
| E5 | Robustness | High-solar hours, evening-ramp hours, top-decile error days, error drift over the test period |
| E6 | Surplus-risk proxy | θ ∈ {0.55, 0.40} × k ∈ {1, 1.5, 2, 3}; risk classification skill |
| E7 | Grid graph | Centrality, electrical distance, bridges; candidate PV/battery buses; candidate reconfigurations checked by AC power flow |
| E8 | Loss / receiving-end prediction | Physics-motivated quadratic Ridge vs RF vs HGB/LightGBM; grouped held-out scenarios; fresh power-flow checks |
| E9 | Optimisation | Baseline (no storage/flex, curtail to constraints) vs forecast-informed LP vs perfect-foresight LP, over all test-period days mapped to the feeder; AC-validated |
| E10 | Weight sensitivity | Objective-weight sweep |

## 7. Success criteria (measurable)

1. Pipeline reruns from raw files with one command. Tests pass.
2. Best model beats seasonal-naive on test MAE for every target/horizon. Where it does not, this is reported as a finding.
3. All test metrics come from the 2025 hold-out only, which is touched once.
4. Grid power flow: 100% of reported scenarios converged. The energy-balance identity holds within 1e-6 MW.
5. Loss/receiving-end models: test R² and MAE reported on held-out scenario groups. Predictions are compared with fresh power-flow runs.
6. Optimisation: zero AC-validated voltage or line violations in reported optimised schedules, or the violations are disclosed. Curtailment, losses, received energy and ramp are reported with absolute and % change vs baseline.
7. Dashboard loads saved artefacts (no retraining) and every metric carries a provenance tag.

## 8. Key assumptions and limitations (summary)

- All-India aggregates cannot show regional transmission congestion, which is a real cause of Indian curtailment.
- "Demand" is demand met. Unserved demand is unobserved.
- The surplus proxy relies on assumed technical minimums and a committed-capacity proxy.
- The 33-bus feeder is a benchmark. Mapping all-India profiles onto it (normalised shapes × feeder ratings) is a scenario device, not a claim about any real Indian feeder.
- Weather features come from archived NWP and may be optimistic for day-ahead use.
