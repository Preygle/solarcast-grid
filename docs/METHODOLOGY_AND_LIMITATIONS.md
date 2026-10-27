# Methodology and Limitations (consolidated)

## Methodology at a glance
| Stage | Method | Detail |
|---|---|---|
| Data | Grid-India all-India SCADA (Mendeley y58jknpgs8 v2), 2021-09 → 2025-06-24 | DATA_AUDIT.md, DATA_PIPELINE.md |
| Cleaning | 10 logged rules: dropouts, bounds, Hampel spikes, frozen telemetry, solar offset; ≤ 2 h interpolation | DATA_PIPELINE.md |
| Duck curve | net load = demand − solar − wind; 3-h evening ramp starting 15–18 h | PHASE_2_REPORT.md |
| Surplus risk | CEA technical-minimum proxy over scenarios θ × k (not observed curtailment) | PROJECT_SPEC.md §4.4 |
| Forecasting | 8 tasks; baselines, Ridge, RF, HistGB, LightGBM, XGBoost, LSTM; chronological train/val/test; selection on validation; conformal PIs | MODEL_METHODOLOGY.md |
| Grid | 33-bus feeder, NetworkX graph, pandapower AC power flow | GRID_MODEL.md |
| Loss / receiving-end prediction | 11,712 simulated operating points; group split by scenario family; fresh power-flow check | PHASE_4_REPORT.md |
| Optimisation | 24-h LP (LinDistFlow, HiGHS) with battery, flexible load and curtailment; 4 cases; AC re-simulation | OPTIMIZATION_METHODOLOGY.md |
| Dashboard | Streamlit + Plotly, reads artefacts; live LP and power flow | USER_GUIDE.md |

## Limitations (all known)

### Data
1. All-India aggregates only. Regional transmission congestion, a major real driver of curtailment in India, is invisible.
2. "Demand" is demand met; load shedding is unobserved.
3. The 2024–25 file is an hourly report with an undocumented aggregation method. Roughness statistics suggest it is comparable with the earlier data, but this is not proven.
4. Cleaning thresholds are judgement calls (all logged). Some residual anomalies remain (e.g. a 19 GW midday demand dip on 2021-11-08).
5. The generation mix (thermal, hydro, nuclear, gas) is available only to 2023-12, so the surplus proxy cannot be computed for 2024–25.
6. No holiday calendar; weather appears only in a labelled experiment, using archived NWP that is optimistic for day-ahead use.

### Forecasting
7. Test-period levels (2025) exceed the training range, which degrades interval coverage (68–82% vs nominal 80%). The level-ratio transform mitigates the problem but does not remove it.
8. Day-ahead solar without weather beats seasonal-naive by only 1.4%, and is worse in the highest-solar hours.
9. Bounded hyper-parameter searches; minimal LSTM search, so its poor ranking is evidence about this configuration rather than about LSTMs in general.
10. A single chronological split (no rolling-origin cross-validation), so the results carry period-specific variance.

### Surplus proxy
11. Committed thermal capacity is proxied by the daily maximum and must-run hydro by the daily minimum. θ is taken from the CEA regulation as reported by secondary sources.

### Grid and receiving-end prediction
12. Benchmark feeder, not India's grid. Line ratings, the voltage band, PV and battery sites and the export limit are stated assumptions.
13. All four PV sites share one all-India shape; PV and battery run at unity power factor.
14. Receiving-end power is an identity of net injection and losses, so its high R² is largely structural; loss prediction (R² ≈ 0.97) is the substantive part.
15. The simulated-label models only show that the learner reproduces the simulator. They say nothing about accuracy against real metering.

### Optimisation
16. LP with LinDistFlow, not AC-OPF. Losses use a convex proxy that underestimates AC losses. Every schedule is AC-validated, but the LP does not optimise AC losses exactly.
17. The ramp objective (hour-to-hour changes) is misaligned with the 3-hour evening ramp metric; the sensitivity analysis shows a non-monotonic response.
18. Baseline and rule-based curtailment use actual PV (reactive real-time control). Only battery and flex decisions are made on forecasts.
19. Battery degradation is a linear throughput cost; there is no calendar ageing and no price signal (no cost data were used).

### Software and environment
20. PuLP is pinned to 3.3.2 (the 4.x API changed). Random Forest artefacts (1.3 GB) are not in git and must be regenerated with `scripts/train_models.py`.

## Bugs found during development (and fixed)
| Bug | Effect | How it was found | Fix |
|---|---|---|---|
| Day-ahead leakage guard used the shortest instead of the longest lead | would have accepted a leaking 24-h lag | unit test | guard requires lag ≥ 38 h |
| Night solar re-filled by interpolation | non-zero night solar | data integrity test | zero-fill all night hours |
| Trees could not extrapolate to 2025 solar levels | day-ahead solar test MAE 2,621 vs val 1,178 MW | pilot run | scale-free level-ratio wrapper |
| Flex defined as a fraction of load | 167/167 forecast-informed realisations infeasible | full optimisation run | absolute-MW energy-neutral shift + regression test |
| Inference index gap (history truncated at τ, then joined to target hours) | lag features misaligned for single-day forecasts; affected the dashboard and the last batch day | dashboard result contradicted batch result | contiguous index + test that the interface reproduces stored test predictions |
