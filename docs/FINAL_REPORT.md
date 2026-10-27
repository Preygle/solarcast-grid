# SolarCast-Grid: Predictive Analysis and Graph-Based Optimisation for Solar Energy Utilisation and Power Delivery

## 1. Abstract
Rising solar generation deepens the midday trough and steepens the evening ramp of net load (the "duck curve"). This creates surplus-solar risk and balancing challenges. We build an end-to-end, reproducible pipeline on real Grid-India all-India SCADA data (September 2021 – June 2025, 33,432 cleaned hourly observations).

Over the period, India's mean evening 3-hour net-load ramp grew from 20.7 GW (2023) to 33.2 GW (H1 2025). We forecast solar, demand, wind and net load at hour-ahead and day-ahead horizons, comparing naive baselines, Ridge, Random Forest, HistGradientBoosting, LightGBM, XGBoost and an LSTM under one chronological protocol. Gradient-boosted trees were selected for 6 of 8 tasks. On a held-out 2025 test period they beat the best naive baseline by 16.6–67.7%, except day-ahead solar (+1.4%), where weather information is needed.

We represent a benchmark 33-bus distribution feeder as a graph and simulate 11,712 AC power flows. Learned models predict technical losses on held-out scenario families with R² 0.974. A forecast-informed 24-hour LP (battery, flexible load, curtailment, linearised network constraints) is re-validated by AC power flow on 167 test days. Against a curtailment-only baseline, it reduces curtailment by 90.5%, technical losses by 17.0% and the feeder's evening ramp by 25.6%, with zero constraint violations. All grid results are simulated on a benchmark network, not measured on India's grid.

## 2. Introduction and background
Solar generation lowers midday net load and leaves conventional generation to meet a steep evening rise. The term "duck chart" originates from CAISO's 2013 analysis. Denholm et al. (NREL, 2015) showed that under business-as-usual flexibility, solar near 20% of annual energy could push marginal curtailment above 30%, and that storage and flexible distributed resources mitigate this. In India, CEA's 2023 regulations require coal units to operate flexibly down to a 40% minimum (55% interim). This defines how far conventional output can be backed down before solar must be curtailed.

## 3. Problem statement
- (A) Predict solar and demand, identify surplus-solar risk, and use the forecasts to improve solar utilisation.
- (B) Predict technical losses and power received at a designated receiving bus.
- (C) Use graph analysis with power flow to find grid-level improvements.

## 4. Aim, research question, objectives
**Research question:** can forecasts learned from real Grid-India data inform a constrained schedule that measurably reduces curtailment and losses on a physically simulated feeder?

Objectives: see `PROJECT_SPEC.md` §2. Predictive analytics is the primary contribution.

## 5. Related work
- Forecasting: probabilistic load forecasting (Hong & Fan 2016); PV forecasting across horizons (Antonanzas et al. 2016); forecast-accuracy measures and the problems of MAPE near zero (Hyndman & Koehler 2006); gradient boosting (LightGBM, Ke et al. 2017); LSTM (Hochreiter & Schmidhuber 1997).
- Grids as complex networks: Pagani & Aiello (2013).
- The 33-bus feeder and loss-minimising reconfiguration: Baran & Wu (1989).
- Power-flow tooling: pandapower (Thurner et al. 2018).

Full list with verified identifiers: `references.md`.

## 6. Data and preprocessing
Primary data: Mendeley y58jknpgs8 v2 (CC BY 4.0), all-India NLDC SCADA, credited to Grid-India/NERLDC. The audit (`DATA_AUDIT.md`) found three formats: 5-min (2021–22), 10-s (2022–23) and hourly (2024–25). It also found dropouts encoded as zeros, spikes up to 466,674 MW, frozen telemetry and solar metering offsets. A ten-rule logged cleaning pipeline (`DATA_PIPELINE.md`) produces the hourly table, interpolating only gaps ≤ 2 h. Candidate portals (ICED, NPP, PVDAQ) were audited and rejected with reasons. No public source provides Indian line topology or receiving-end metering.

## 7. Exploratory analysis and duck-curve observations
- The daily net-load minimum shifted from 03:00 (2021) to 13:00 (2022 onward).
- Evening 3-hour ramps averaged 21.0 / 20.7 / 25.4 / 33.2 GW in 2022 / 2023 / 2024 / H1 2025, with a 50.6 GW maximum.
- The duck shape is strongest pre-monsoon (2025: ~140 GW at 13:00 rising to ~196 GW at 19:00).
- Surplus-risk proxy: no surplus hours at the observed 2021–23 fleet. Risk appears only at ≥ 2× solar with a 55% technical minimum, and lowering the minimum to 40% removes most of it.

Figures: `results/eda/`.

## 8. Predictive methodology
Eight tasks (4 targets × 1-h / day-ahead). Day-ahead forecasts are issued at 10:00 on D−1, and every feature is computed from data available at that issue time (enforced lag ≥ 38 h; perturbation tests at both the feature and inference level). Splits: train to 2024-06, validation 2024-07–12, test 2025-01–06-24. Bounded search on validation, refit on train+val, test used once.

A key design decision is a **scale-free level-ratio target**, because 2025 solar peaks exceed all training values and tree models cannot extrapolate (pilot: test MAE 2,621 → 1,266 MW after the fix). Intervals are split-conformal. Details: `MODEL_METHODOLOGY.md`.

## 9. Model comparison and evaluation
`MODEL_EVALUATION.md` gives the full table. Summary:
- LightGBM / XGBoost / HistGB are statistically close and best on 6 of 8 tasks; Ridge wins day-ahead wind; the LSTM is never selected.
- Hour-ahead errors are 2–5× smaller than day-ahead.
- Day-ahead solar barely beats seasonal-naive and loses in the top-10% solar hours (−12.7%). Archived NWP weather reduces its error by 10.5% (an optimistic upper bound).
- The same-hour 7-day mean is the dominant feature; calendar-only models are 3–5× worse.
- Interval coverage is 68.5–81.5% against a nominal 80%, reflecting distribution shift.

## 10. Grid graph construction and analysis
The 33-bus, 12.66 kV feeder has 37 lines including 5 tie switches, built as a NetworkX graph with impedance, rating and tie attributes:
- 5 loops; 50,751 radial configurations (matrix-tree theorem).
- Line 0 is the only line with no alternative path.
- Resistance distance identifies the lateral ends (buses 17, 21, 24, 32) as PV sites.

Graph candidates were judged by AC power flow. Branch exchange gives −31.1% peak losses, while the graph-only minimum spanning tree *raises* losses and breaches 0.90 pu. For battery siting, which graph metric correlates with the AC benefit depends on the condition (Spearman 0.76 at peak for loss moment; 0.96 at midday for resistance distance). Details: `GRID_MODEL.md`.

## 11. Power-flow and receiving-end prediction methodology
- Receiving bus = bus 0 (grid supply point); `P_recv = −P_ext_grid`.
- Losses = Σ line (p_from + p_to). The energy balance is asserted to 1e-6 MW on every run.
- 500 scenario families (observed day × configuration × PV capacities × battery profile × load noise) give 11,692 converged points.
- Group split by family; leakage guard on features; fresh power-flow check on new seeds.
- Loss MAE 12.2 kW (R² 0.974) against 28.1 kW for a physics-motivated quadratic Ridge.
- Received-power R² 0.9996 is largely structural (received = net injection − losses).

## 12. Optimisation formulation
A 24-hour LP (PuLP + HiGHS) on LinDistFlow with:
- PV curtailment, battery charge/discharge/state of charge (1.5 MW / 6 MWh, η = 0.95, cyclic) and energy-neutral flexible load (±10%);
- voltage 0.905–1.045 pu, line and export (1 MW) limits;
- objective: curtailed MWh + loss-proxy MWh + 0.05·ramp + 0.005·throughput.

The forecast-informed plan is fixed and realised on the actual day, with soft slacks reported as violations. Every hour is AC re-simulated. Details: `OPTIMIZATION_METHODOLOGY.md`.

## 13. Baseline-versus-optimised results
Over 167 days, all cases were Optimal with 0 AC violations.

| Metric | Baseline | Rule-based | Forecast-informed | Perfect foresight |
|---|---|---|---|---|
| Curtailment (MWh) | 967.9 | 578.8 | **91.7** | 41.5 |
| Solar utilisation | 86.4% | 91.9% | **98.7%** | 99.4% |
| Losses (MWh) | 403.3 | 392.1 | **334.9** | 336.6 |
| Mean evening ramp (MW) | 3.63 | 3.27 | **2.70** | 2.74 |
| Export to upstream (MWh) | 990.4 | 1,013.0 | **676.5** | 707.5 |

Forecast-informed scheduling captures 93.6% of the perfect-foresight curtailment benefit. Gross export to the upstream utility falls by 31.7%, because surplus is shifted to local evening demand. This trade-off is reported rather than claimed as a gain. Sensitivity: curtailment is 45.9 / 14.0 / 4.4 MWh for 3 / 6 / 12 MWh batteries (20-day sample). The ramp-weight response is non-monotonic, which reveals a misalignment between the ramp objective and the ramp metric.

## 14. Discussion
Forecast quality matters, but less than one might expect for this decision. Day-ahead solar forecasts only marginally better than persistence still deliver most of the achievable benefit. A battery schedule is robust to moderate forecast error when real-time curtailment absorbs the residual, and the main driver of benefit is having storage and flexibility at all.

The forecasting results show where the effort should go:
- solar day-ahead needs weather inputs;
- non-stationarity (capacity growth) must be handled explicitly;
- intervals need recalibration under drift.

On the grid side, graph theory is useful for search-space reduction and candidate ranking, but electrical evaluation is indispensable: the MST counter-example shows a graph-optimal topology can be electrically worse.

## 15. Limitations
See `METHODOLOGY_AND_LIMITATIONS.md` (20 itemised limitations and 5 bugs found and fixed). The most important: all-India aggregates hide regional congestion; the surplus metric is a proxy; the grid is a benchmark feeder driven by mapped shapes; LinDistFlow is not AC-OPF; the Indian receiving-end and loss figures are not measured.

## 16. Conclusion and future work
The project delivers a validated forecasting stack on real Indian data, a reproducible duck-curve analysis, and simulation-based evidence that forecast-informed storage and flexibility scheduling can largely eliminate curtailment on a constrained feeder while reducing technical losses.

Future work:
- operational NWP inputs at issue time and regional (state-level) data;
- rolling-origin evaluation and adaptive interval calibration;
- a real feeder model with metered receiving-end data;
- AC-OPF or iterative LP–power-flow coupling;
- price-aware objectives and battery ageing.

## 17. References
See `references.md` (datasets D1–D5; R1–R3; F1–F5; G1–G4; software versions).
