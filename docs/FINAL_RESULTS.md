# Final Results (key measured findings)

Every number below is copied from a committed results file. Provenance class in brackets.

## 1. Data and duck curve [OBSERVED / DERIVED]
- 33,432 hourly rows (2021-09-01 → 2025-06-24) from 4,075,224 raw SCADA samples. After cleaning, 65 demand hours and 143 solar hours remain missing (`results/eda/preprocessing_log.json`).
- The daily net-load minimum moved from night (03:00, 2021) to midday (13:00, 2022 onward).
- Mean evening 3-hour net-load ramp: 20.7 GW (2023) → 25.4 GW (2024) → **33.2 GW (H1 2025)**. Maximum 50.6 GW (2025-03-05). Mean solar share of demand energy: 6.4% → 9.3% (`duck_yearly_summary.csv`).
- Surplus-risk proxy: **zero surplus hours at the observed fleet** for θ = 0.55 or 0.40. With θ = 0.55, surplus appears at 2× solar (0.20% of solar energy) and grows to 9.15% at 3× solar. With θ = 0.40, it is 1.42% at 3× solar (`surplus_proxy_scenarios.csv`). This is an assumption-driven indicator, not observed curtailment.

## 2. Forecasting [FORECAST vs OBSERVED, test 2025-01-01 → 06-24]
| Task | Selected (val) | Test MAE (MW) | R² | vs best baseline |
|---|---|---|---|---|
| Solar, day-ahead | HistGB | 1,277 | 0.988 | +1.4% |
| Solar, 1 h | LightGBM | 559 | 0.998 | +47.0% |
| Demand, day-ahead | XGBoost | 4,631 | 0.887 | +33.9% |
| Demand, 1 h | LightGBM | 1,427 | 0.989 | +67.6% |
| Wind, day-ahead | Ridge | 2,040 | 0.783 | +24.3% |
| Wind, 1 h | LightGBM | 389 | 0.992 | +36.5% |
| Net load, day-ahead | XGBoost | 6,488 | 0.830 | +16.6% |
| Net load, 1 h | LightGBM | 2,016 | 0.982 | +67.7% |

- Gradient-boosted trees were selected for 6 of 8 tasks. The LSTM was never selected.
- Weather (archived NWP, optimistic) improves day-ahead solar by 10.5%, demand by 2.8% and net load by 2.0%.
- 80% conformal intervals cover 68.5–81.5% on test (under-coverage from distribution shift).

## 3. Grid graph and power flow [SIMULATED]
- Base case reproduces the published 0.2027 MW losses.
- 50,751 radial configurations.
- Branch-exchange reconfiguration cuts peak losses by 31.1%; the graph-only MST raises them by 1.4%–33.7%.

## 4. Losses and receiving-end power [SIMULATED labels, held-out families]
- Losses: LightGBM test MAE 12.2 kW, R² 0.974.
- Received power at bus 0: HistGB MAE 21.1 kW, R² 0.9996 (largely structural).
- Loss %: MAE 0.27 points, R² 0.971.
- Fresh power-flow check on 886 new points: loss MAE 13.4 kW, received MAE 22.0 kW.

## 5. Optimisation [OPTIMISED, AC-validated, 167 test days, 0 violations]
| vs baseline | Rule-based | Forecast-informed | Perfect foresight |
|---|---|---|---|
| Curtailment | −40.2% | **−90.5%** (967.9 → 91.7 MWh) | −95.7% |
| Solar utilisation | 91.9% | **98.7%** (from 86.4%) | 99.4% |
| Technical losses | −2.8% | **−17.0%** (403.3 → 334.9 MWh) | −16.5% |
| Mean daily peak import | −2.2% | **−24.3%** | −26.0% |
| Mean evening 3-h ramp | −9.8% | **−25.6%** | −24.4% |
| Export received by upstream utility | +2.3% | **−31.7%** (trade-off: surplus is used locally) | −28.6% |

- Forecast-informed scheduling achieves 93.6% of the perfect-foresight curtailment reduction.
- The dashboard's live optimiser was verified in the running app on 2025-04-12: curtailment 9.10 → 1.71 MWh and losses 2.28 → 1.94 MWh, identical to the batch evaluation.

## 6. What the results do **not** show
- Nothing here is a measurement of Indian receiving-end power or technical losses.
- The feeder-level benefits depend on the stated assumptions (1 MW export limit, 6 MW PV on a 3.7 MW feeder, 1.5 MW / 6 MWh battery). They are not national estimates.
