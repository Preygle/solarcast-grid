# Grid Model (SIMULATED)

**This is a benchmark network, not India's grid.** No public source audited in Phase 1 provides Indian line-level topology, impedances or receiving-end metering, so every result in this document is SIMULATED with AC power flow.

## Network: Baran & Wu 33-bus feeder (`pandapower.networks.case33bw`)
| Item | Value | Source |
|---|---|---|
| Voltage | 12.66 kV, slack (ext. grid) at bus 0, 1.0 pu | case file |
| Buses / lines | 33 buses; 37 lines (32 in service + 5 normally-open tie lines 32–36) | case file |
| Loads | 32 loads, 3.715 MW + 2.300 Mvar total (base = feeder peak) | case file |
| Line impedances | r, x per case file (lengths 1 km) | case file |
| **Line rating** | **0.25 kA** for every line (≈5.5 MVA) | **assumption**: case file has none |
| **Voltage band** | **0.90 – 1.05 pu** | **assumption**: case file 0.90–1.10; 1.05 is a common DG over-voltage limit |
| **PV plants** | 1.5 MW each at buses **17, 21, 24, 32** (lateral ends) | **assumption**, supported by graph analysis (highest resistance distance) |
| **Battery** | 1.5 MW / 6 MWh at bus **17** | selected by AC-PF evaluation of graph candidates (below) |
| **Export limit** | 1.0 MW reverse power at bus 0 | **assumption** (upstream hosting/protection limit) |
| **Receiving-end bus** | **bus 0**, the grid supply point to the upstream utility | definition |

Validation: base-case AC power flow gives losses of **0.2027 MW** and V_min **0.9131 pu** at bus 17, matching the published base case of this feeder (unit test `test_base_case_matches_published_losses`).

## Definitions (src/power_flow/runner.py)
- `p_loss_mw` = Σ_lines (p_from + p_to): technical I²R losses only. Commercial losses and aggregate T&D losses are outside this model.
- `p_recv_mw` = −P_ext_grid: power received by the upstream utility at bus 0. Negative values mean import.
- `p_sent_mw` = PV + battery discharge + import; `p_delivered_mw` = p_sent − p_loss; `loss_pct` = 100·p_loss/p_sent.
- Checked on **every** power flow: `ΣPV + P_batt − ΣLoad − P_loss − P_recv = 0` within 1e-6 MW. Otherwise an exception is raised. Non-converged runs are flagged and excluded, never imputed.
- Meshed vs radial: all results come from the physical AC power flow. No single-path "received = sent − path loss" assumption is used.

## Graph representation (src/grid/graph.py, NetworkX)
Nodes = buses (base load, PV/battery flags, voltage limits). Edges = lines (r, x, |z| in Ω, rating, tie flag, in-service).

| Metric | Use | Result |
|---|---|---|
| Tree check | the default configuration must be radial | radial (tree) |
| Cycle basis | each tie line closes one loop → reconfiguration search space | 5 independent loops |
| Spanning-tree count (matrix-tree theorem) | size of the radial configuration space | **50,751** |
| Bridges of the meshed graph | lines with no alternative path even with ties closed | line 0 (substation–bus 1) only: the single point of failure |
| Resistance distance to bus 0 | voltage-rise / loss sensitivity of injections | max at bus 17 (11.06 Ω) |
| Impedance-weighted betweenness | buses that carry many supply paths | trunk buses 1–5 |
| Loss moment = downstream load × path resistance | graph proxy for loss reduction from local injection | top 5: buses 5, 28, 12, 9, 29 |

## Graph candidates evaluated by AC power flow
**Reconfiguration** (`results/grid_scenarios/reconfiguration.csv`). Branch exchange: close an open tie, open a line in the created loop, keep the change if AC losses fall.

| Condition | Configuration (open lines) | AC losses (MW) | Δ vs default | V_min (pu) |
|---|---|---|---|---|
| Peak load, no PV | default 32-33-34-35-36 | 0.2027 | — | 0.913 |
| Peak load, no PV | branch exchange 6-8-13-31-36 (447 PF evaluations) | **0.1396** | **−31.1%** | 0.938 |
| Peak load, no PV | min. spanning tree on r (pure graph) | 0.2056 | +1.4% | **0.898 (violates 0.90)** |
| Midday PV (0.8 load, 4×1.35 MW) | default | 0.1637 | — | — |
| Midday PV | branch exchange 5-8-34-35-36 | 0.1486 | −9.2% | — |
| Midday PV | min. spanning tree on r | 0.2189 | +33.7% | — |

In 1-based branch numbering, the peak-load optimum (lines 6-8-13-31-36) is branches 7-9-14-32-37. **The minimum-resistance spanning tree, a purely graph-theoretic choice, increases losses and violates the voltage limit.** Graph methods are therefore used to generate candidates, and AC power flow decides.

**Battery siting** (`battery_siting.csv`): 1 MW injection (peak) / absorption (midday PV) at each bus.

| Condition | Best bus | Loss reduction | Spearman(PF benefit, graph metric) |
|---|---|---|---|
| Peak load, discharge | 28 (74.4 kW) | — | loss moment **0.76**, resistance distance 0.64, betweenness 0.19 |
| Midday PV, charge | 17 (65.9 kW) | — | resistance distance **0.96**, loss moment 0.14, betweenness −0.49 |
| Sum of both | **17** (122.8 kW) | selected | — |

Which graph metric is useful depends on the operating condition. Loss moment ranks discharge locations well at peak; resistance distance ranks absorption locations near PV well. The battery's purpose here is absorbing surplus PV, so bus 17 (co-located with a PV plant) was chosen.

## Scenario dataset for Problem B
See `src/simulation/scenarios.py` and `docs/PHASE_4_REPORT.md`. Each family = one observed day (all-India demand/solar shapes) × one of 7 radial configurations × random PV capacity per site (0–1.5× rating) × random battery sinusoid (±0–2 MW) × per-bus load noise (lognormal σ = 0.10) × feeder load multiplier (0.7–1.3). Every row records family, seed, source day, configuration, inputs, solver outputs and the `label_source` tag.
