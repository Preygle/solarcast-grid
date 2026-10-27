# Optimisation Methodology

**Data classes.** Feeder = SIMULATED (33-bus benchmark). Profile shapes = OBSERVED all-India data (actual case) or FORECAST (Phase 3 day-ahead models). Decisions = OPTIMISED. All reported losses, voltages and energies come from **AC power-flow re-simulation** of the realised schedule.

## Problem addressed
Midday solar on the feeder can exceed what the grid can absorb. That limit is set by the reverse-power (export) limit at the receiving bus, the 1.05 pu over-voltage limit and the line ratings. The excess must be curtailed unless storage or flexible demand absorbs it. The aim is to cut avoidable curtailment and losses, and flatten the feeder's evening ramp, without violating constraints.

## Mapping all-India profiles to the feeder (scenario device)
`load_scale_t = demand_t / max(demand, train+val)`; `pv_pu_t = min(1, solar_t / S_ref)`, where `S_ref` is the rolling 30-day maximum of solar known at the day-ahead issue time (D−1 09:00). The same `S_ref` is used for forecast and actual. PV per site = 1.5 MW × `pv_pu` at buses 17, 21, 24, 32 (6 MW total vs a 3.715 MW feeder peak load). All four sites share the all-India shape, which is a simplification.

## Decision variables (per hour t, 24 h)
| Variable | Meaning | Bounds |
|---|---|---|
| `u_kt` | PV used at site k (MW); curtailment `c_kt = a_kt − u_kt` | 0 ≤ u ≤ available a_kt |
| `ch_t`, `dis_t` | battery charge / discharge (MW) | 0 … P_max (1.5 MW) |
| `soc_t` | state of charge (MWh) | 10%–90% of 6 MWh |
| `d_t` | flexible-load shift (MW, feeder total), spread over loads in proportion to base load | \|d_t\| ≤ 10% of planned load |
| `P_jt`, `Q_jt`, `v_jt` | LinDistFlow branch flows and squared voltages | see constraints |

## Constraints
1. **Power balance (LinDistFlow).** `P_j = withdrawal_j + Σ_children P_c`, with the same form for Q. Withdrawal = load + d·share − PV + charge − discharge at the battery bus.
2. **Voltage.** `v_j = v_i − 2(r_ij P_j + x_ij Q_j)` (per-unit, 1 MVA / 12.66 kV base). Bounds are (0.90 + 0.005)² ≤ v ≤ (1.05 − 0.005)², with a 0.005 pu margin for linearisation error. `v_0 = 1`.
3. **Line limits.** |P_j| ≤ 0.9·√3·12.66 kV·0.25 kA (≈4.93 MW). The 0.25 kA rating is an assumption; the case file has no ratings.
4. **Export limit.** Reverse power at the receiving bus ≤ 1.0 MW (assumed upstream hosting / protection limit).
5. **Battery.** `soc_t = soc_{t−1} + η·ch_t − dis_t/η`, with η = 0.95 one-way. `soc_23 ≥ soc_init` (no free energy from the initial state).
6. **Flexible load.** Energy-neutral over the day: Σ d_t = 0. It is defined in MW rather than as a fraction of load, so a plan made on forecast loads stays energy-neutral when it is applied to the actual day. A first implementation used fractions; that version made every forecast-informed realisation infeasible (167/167 days), which is how the issue was found.

## Objective (MWh-equivalent; weights documented and swept)
`min  1.0·Σ curtailment  +  1.0·Σ loss_proxy  +  0.05·Σ|head_t − head_{t−1}|  +  0.005·Σ(ch + dis)`
- Curtailed energy and loss energy are both lost energy, so they get equal weight (no arbitrary normalisation is needed).
- `loss_proxy_jt ≥ r_j(±2p₀P_jt − p₀²)` for tangent points p₀ ∈ {0.25 … 5} MW. This is a convex outer approximation of r·P². It ignores reactive-power losses, so it **underestimates** AC losses; AC re-simulation reports the true value.
- The ramp term flattens the feeder's import profile (a local duck curve).
- The throughput term is a small degradation cost that stops pointless cycling.
- Solver: PuLP 3.3.2 + HiGHS. Every schedule's solver status is recorded.

## Cases compared on identical days
| Case | Planning information | Battery / flex | Realisation |
|---|---|---|---|
| baseline | — | none | curtailment-only LP on actual day (curtail exactly as limits require) |
| rule_based | actual (reactive) | charge from surplus above export limit; discharge evenly 17–22 h | curtailment LP with battery fixed |
| forecast_informed | **day-ahead forecasts** (selected Phase 3 models) | LP schedule | battery & flex **fixed** from the plan; curtailment re-optimised on the actual day; constraint slacks (penalty 1000) reported as violations |
| perfect_foresight | actual | LP schedule | same (upper bound) |

## Validation
Each realised hour is re-run with pandapower AC power flow. We report losses, receiving-end energy, voltages, loadings and limit violations. The LinDistFlow export constraint excludes losses, so AC export can differ slightly from the LP value. The difference is reported (`export_violation_hours` uses a 1 kW tolerance).

## Limitations
- The linearised network model checks voltage and losses only through AC re-simulation; this is not an AC-OPF.
- Reactive power of PV and the battery is fixed at zero (unity power factor).
- A single representative feeder is driven by all-India shapes; regional diversity is ignored.
- Battery degradation is only a linear throughput cost.
