# Phase 4 Report — Grid Graph, Power Flow, Loss / Receiving-End Prediction, Optimisation

**Status: PASSED** (self-reviewed; the user asked for autonomous execution). **All results in this phase are SIMULATED on the 33-bus benchmark feeder.** They are not measurements of the Indian grid.

## Work completed
| File | Purpose |
|---|---|
| `src/grid/network.py` | case33bw + documented assumptions (ratings, voltage band, PV/battery sites, export limit, receiving bus) |
| `src/grid/graph.py` | NetworkX graph, node/edge metrics, loops, radiality, spanning-tree count, branch-exchange neighbours |
| `src/power_flow/runner.py` | AC power flow, loss / receiving-end accounting, energy-balance assertion, violation flags |
| `src/simulation/scenarios.py` | all-India → feeder profile mapping; labelled scenario generator with full metadata |
| `src/receiving_power/model.py` | leakage-guarded features, models, group split, metrics |
| `src/optimization/lp.py`, `src/optimization/evaluate.py` | 24-h LinDistFlow LP (PuLP/HiGHS); 4 cases; AC validation |
| `scripts/run_grid_analysis.py`, `run_receiving_power.py`, `run_optimization.py` | reproducible runs |
| `results/grid_scenarios/*` | graph metrics, reconfiguration, battery siting, scenarios (11,712 rows), fresh PF check, optimisation daily / sensitivity / meta |
| `results/grid_model_comparison.csv`, `results/optimization_comparison.csv` | required comparison tables |
| `results/figures/grid_*.png`, `optimization_example_days.png` | graph, siting, predicted vs simulated, model comparison, example days |
| `models/receiving_power/*.joblib` | selected loss / receiving-end / loss-% models |
| `docs/GRID_MODEL.md`, `docs/OPTIMIZATION_METHODOLOGY.md` | documentation |
| `tests/test_grid_optimization.py` | 8 tests (published base case, balance & sign conventions, graph structure, LP battery physics, flex neutrality, AC validation of LP schedules, plan realisable on a different day, rule bounds) |

## Evidence

### Graph + power flow (Problem C)
- Base case reproduces the published 33-bus result: losses 0.2027 MW, V_min 0.9131 pu (unit-tested).
- 5 loops; **50,751** radial configurations (matrix-tree theorem, tested); line 0 is the only line with no alternative path.
- Branch-exchange reconfiguration, scored by AC power flow, reduces peak-load losses by **31.1%** (0.2027 → 0.1396 MW; opens lines 6-8-13-31-36) and midday-PV losses by **9.2%**.
- The impedance-weighted minimum spanning tree (graph-only) **increases** losses by 1.4% at peak and 33.7% at midday, and breaches 0.90 pu. Graph metrics generate candidates; AC power flow decides.
- Battery siting: which graph metric predicts the AC benefit depends on the condition. At peak, loss moment has Spearman 0.76; at midday absorption, resistance distance has Spearman 0.96. Bus 17 was selected (total 122.8 kW over both conditions).

### Loss and receiving-end prediction (Problem B)
- **Simulation.** 500 scenario families × 24 h = 11,712 AC power flows (388 s). 11,692 converged; the 20 non-converged were excluded and recorded. Maximum energy-balance residual 5.3e-15 MW.
- **Split.** GroupShuffleSplit by family (train 70% / val 15% / test 15%); no family is shared between splits (asserted).
- **Held-out test results:**

| Target | Quadratic Ridge (physics baseline) | Random Forest | HistGB | LightGBM | Selected (val) |
|---|---|---|---|---|---|
| Losses [MW] MAE / R² | 0.0281 / 0.908 | 0.0199 / 0.933 | 0.0130 / 0.970 | **0.0122 / 0.974** | LightGBM |
| Received at bus 0 [MW] MAE / R² | 0.0281 / 0.9994 | 0.0337 / 0.9989 | **0.0211 / 0.9996** | 0.0268 / 0.9994 | HistGB |
| Loss % MAE / R² | 0.674 / 0.866 | 0.404 / 0.933 | 0.294 / 0.965 | **0.273 / 0.971** | LightGBM |

- **Fresh power-flow check.** 40 new families from an unseen seed (886 converged points), each compared with its own new AC power flow: loss MAE 13.4 kW (R² 0.971), received-power MAE 22.0 kW (R² 0.9995), loss-% MAE 0.30 points (R² 0.970).
- **Interpretation.** Received power equals net injection (a known input) minus losses. Its very high R² is therefore largely structural, and the substantive prediction problem is the loss term (R² ≈ 0.97). The quadratic Ridge has identical MAE for losses and received power, which confirms that it learns the linear part exactly.

### Optimisation (Problem A)
167 test days (2025-01-01 → 2025-06-24 with complete data). **All 4 cases Optimal on all 167 days.** AC validation found **0 voltage, 0 thermal and 0 export-limit violation hours**, and the LP slack was 0. Day-ahead forecast MAE on these days: solar 1,304 MW, demand 4,661 MW.

| Metric (AC-validated, sum over 167 days) | Baseline | Rule-based | **Forecast-informed** | Perfect foresight |
|---|---|---|---|---|
| PV available [MWh] | 7,140.2 | 7,140.2 | 7,140.2 | 7,140.2 |
| Curtailment [MWh] | 967.9 | 578.8 (−40.2%) | **91.7 (−90.5%)** | 41.5 (−95.7%) |
| Solar utilisation [%] | 86.4 | 91.9 | **98.7** | 99.4 |
| Technical losses [MWh] | 403.3 | 392.1 (−2.8%) | **334.9 (−17.0%)** | 336.6 (−16.5%) |
| Import from upstream [MWh] | 7,234.8 | 6,895.1 (−4.7%) | **6,067.5 (−16.1%)** | 6,061.9 |
| Export received by upstream utility [MWh] | 990.4 | 1,013.0 | **676.5 (−31.7%)** | 707.5 |
| Mean daily peak import [MW] | 3.26 | 3.19 | **2.47 (−24.3%)** | 2.42 |
| Mean evening 3-h ramp [MW] | 3.63 | 3.27 (−9.8%) | **2.70 (−25.6%)** | 2.74 |
| Battery throughput [MWh] | 0 | 740 | 1,778 | 2,009 |

Interpretation:
- Forecast-informed scheduling captures **93.6%** of the curtailment reduction achievable with perfect foresight ((967.9 − 91.7) / (967.9 − 41.5)).
- **Energy exported to the upstream utility falls.** Absorbed midday surplus is used locally in the evening instead of being exported at the limit. The feeder's net exchange improves (import −16.1%), but the gross export metric gets worse. Whether that is desirable depends on the offtaker's objective, so it is reported as a trade-off, not an improvement.
- Forecast-informed AC losses are marginally *lower* than with perfect foresight (334.9 vs 336.6 MWh). The LP minimises a loss *proxy* (which underestimates AC losses, e.g. 1.21 vs 2.28 MWh on 2025-04-12), not AC losses, so the two schedules differ in ways the proxy does not rank perfectly.

**Sensitivity** (forecast-informed, 20 evenly spaced test days, `optimization_sensitivity.csv`):
- Battery 3 / 6 / 12 MWh (power = E/4) → curtailment 45.9 / 14.0 / 4.4 MWh.
- Loss weight 0 → 3: curtailment 10.5 → 14.8 MWh and losses 40.7 → 39.1 MWh (expected trade-off).
- Ramp weight 0 / 0.05 / 0.2 / 1.0 → mean evening ramp 2.70 / 2.74 / 2.91 / 2.77 MW. This is **not monotonic**. The ramp term penalises every hour-to-hour change in import, not specifically the 15–21 h three-hour ramp metric, and it competes with the other terms. The metric and the objective term are not aligned. This is a documented limitation.

## Key decisions
- The benchmark feeder was used because real Indian topology is unavailable; every assumption is listed in GRID_MODEL.md.
- Graph analysis is used for candidate generation, with AC power flow as the judge (demonstrated by the MST counter-example).
- LP with LinDistFlow plus AC re-simulation instead of AC-OPF: tractable and auditable, with the approximation error exposed.
- Bug found and fixed during this phase: flexible load defined as a *fraction* of load made every forecast-informed realisation infeasible (167/167 days). It is now an absolute MW shift that is energy-neutral by construction, and a regression test was added.

## Limitations
- One benchmark feeder driven by all-India shapes; all four PV sites share one profile.
- PV and battery at unity power factor; the battery is modelled as power-only with linear degradation cost.
- The LinDistFlow loss proxy underestimates AC losses, and the ramp objective is misaligned with the ramp metric.
- Rule-based and baseline curtailment use the actual day's PV (reactive real-time curtailment), while battery plans use forecasts.

## Acceptance criteria
| Criterion | Met |
|---|---|
| Documented grid represented and analysed as a graph | Yes |
| Valid power-flow scenarios executed | Yes (11,712 runs; convergence recorded) |
| Well-defined losses / receiving-end power with units | Yes (definitions + energy-balance assertion) |
| Prediction on held-out scenarios | Yes (group split + fresh PF check) |
| Optimisation with documented variables / objective / constraints | Yes |
| Baseline vs optimised under consistent scenarios | Yes (identical 167 days, 4 cases) |
| Improvements supported by computed results | Yes (AC-validated; negative findings reported) |
| Actual vs simulated distinguished | Yes |
