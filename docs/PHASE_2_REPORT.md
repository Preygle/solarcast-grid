# Phase 2 Report — Data Engineering, EDA, Duck Curve, Baselines

**Status: PASSED** (self-reviewed; the user asked for autonomous execution).

## Work completed
| File | Purpose |
|---|---|
| `src/data/load_raw.py`, `src/data/preprocess.py` | parsers for all three source layouts; 10-step documented cleaning; hourly table |
| `src/features/build.py`, `src/features/splits.py` | issue-time-aware, leak-safe features; chronological splits |
| `src/analysis/duck_curve.py`, `src/analysis/surplus.py`, `src/analysis/eda.py` | duck metrics, surplus proxy, figures |
| `src/models/baselines.py`, `src/evaluation/metrics.py` | naive forecasts and metrics |
| `scripts/build_dataset.py`, `scripts/run_eda.py`, `scripts/run_baselines.py` | one-command reruns |
| `tests/test_data_pipeline.py`, `tests/test_features_baselines.py` | 19 tests |
| `data/processed/hourly.parquet` | 33,432 hourly rows |
| `results/eda/*` | 12 figures, summary statistics, preprocessing log, duck/surplus tables |
| `results/baseline_metrics.csv` | baseline metrics (val + test) |
| `docs/DATA_PIPELINE.md` | pipeline, data dictionary, conventions |

## Evidence

**Tests:** `python -m pytest -q` → **19 passed**. These include a perturbation leakage test: values after the issue time are multiplied by 1.37 and the features must not change. Writing the tests exposed two bugs, which are now fixed. (1) The day-ahead leakage guard checked lags against the *shortest* lead (15 h) instead of the longest (38 h), so it would have accepted a leaking 24-h lag. (2) Night-time solar could be re-filled by interpolation.

**Data quality (actual counts).** 4,075,224 raw samples → 33,432 hours. Remaining NaN: demand 65 h, solar 143 h, wind 295 h. Interpolated (≤ 2 h gaps): demand 440 h, solar 475 h. Main issues found: zero-coded dropouts, spikes up to 466,674 MW, frozen telemetry (e.g. 64,301 MW repeated at night), and a per-day solar night offset of up to 2,028 MW.

**Duck curve (DERIVED from OBSERVED):**

| Year | days | min net load (MW) | mean evening 3-h ramp (MW) | p95 ramp | max ramp | % days duck-shaped | modal hour of daily min |
|---|---|---|---|---|---|---|---|
| 2021 (Sep–Dec) | 119 | 101,521 | 24,075 | 29,742 | 35,123 | 100 | 03 |
| 2022 | 353 | 98,410 | 20,994 | 32,290 | 42,026 | 96 | 13 |
| 2023 | 358 | 112,573 | 20,705 | 29,250 | 37,657 | 96 | 13 |
| 2024 | 334 | 116,588 | 25,367 | 36,241 | 41,187 | 99 | 13 |
| 2025 (Jan–Jun) | 167 | 119,714 | 33,161 | 42,300 | 50,647 | 100 | 13 |

Findings:
- From 2022 the daily net-load minimum moved from night (03:00) to 13:00.
- The mean evening ramp rose by ~60% between 2023 and H1 2025, while mean solar share of demand energy rose from 6.4% to 9.3%.
- "Duck-shaped" is defined here as a midday (11–14 h) mean below both the morning (6–9 h) and evening (18–21 h) means. This is a weak criterion: the Indian morning demand peak makes most days qualify even with modest solar. The ramp size is the more informative metric.
- In 2021 the daily minimum was at night, so the midday trough was not yet the lowest point of the day.

**Surplus-risk proxy** (2021-09 → 2023-12 generation mix, 20,330 hours; assumption-driven):

| θ (tech-min) | k (solar ×) | surplus hours | days | surplus GWh | % of scaled solar |
|---|---|---|---|---|---|
| 0.55 | 1.0 / 1.5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 0.55 | 2.0 | 199 | 74 | 809 | 0.20 |
| 0.55 | 2.5 | 1,545 | 424 | 13,516 | 2.65 |
| 0.55 | 3.0 | 3,631 | 776 | 56,075 | 9.15 |
| 0.40 | 2.5 | 99 | 36 | 461 | 0.09 |
| 0.40 | 3.0 | 1,030 | 312 | 8,699 | 1.42 |

At the observed fleet (k = 1) the proxy shows **no** surplus hours under either θ, which is consistent with the absence of system-wide curtailment in the 2021–23 aggregate data. Surplus risk appears only for solar build-out of ≥ 2× under today's demand and thermal fleet. Lowering the thermal technical minimum from 55% to 40% removes most of it. Sanity check: observed thermal never fell below θ × daily max (0 of 20,435 hours), so the floor is consistent with observed dispatch. Regional transmission-limited curtailment cannot be seen in all-India aggregates.

**Baselines (test set 2025-01-01 → 2025-06-24, best baseline per task):**

| Target | Horizon | Best baseline | MAE (MW) | RMSE (MW) | R² | nMAE % |
|---|---|---|---|---|---|---|
| Solar | 1 h | seasonal-naive 24 h | 1,055 | 2,145 | 0.991 | 5.74 |
| Solar | day-ahead | same hour D−1 (t−48) | 1,295 | 2,662 | 0.987 | 7.06 |
| Demand | 1 h | persistence | 4,396 | 5,826 | 0.909 | 2.22 |
| Demand | day-ahead | t−48 | 7,003 | 9,413 | 0.758 | 3.53 |
| Wind | 1 h | persistence | 612 | 800 | 0.981 | 6.61 |
| Wind | day-ahead | same-hour mean 7 d | 2,694 | 3,558 | 0.627 | 29.08 |
| Net load | 1 h | seasonal-naive 24 h | 6,235 | 8,401 | 0.835 | 3.66 |
| Net load | day-ahead | t−48 | 7,781 | 10,121 | 0.761 | 4.56 |

For day-ahead, "persistence" and "seasonal-naive 24 h" are the same forecast (both t−48 h, the latest complete day at issue time). Full table: `results/baseline_metrics.csv`.

## Assumptions and limitations
- IST timestamps (consistent with the solar peak in the 12:00 hour); the aggregation method of the 2024 report is unverified.
- Cleaning thresholds are judgement calls, all logged. A few residual anomalies remain, e.g. a 19 GW midday demand dip on 2021-11-08.
- No holiday calendar; no weather in the baseline features.
- The surplus proxy depends on the committed-capacity and must-run-hydro proxies.

## Acceptance criteria
| Criterion | Met |
|---|---|
| Loading/preprocessing works on selected data | Yes |
| Units, timestamps, missing values, assumptions documented | Yes |
| Duck-curve analysis reproducible | Yes (`scripts/run_eda.py`) |
| Baselines from genuine test observations | Yes |
| Leakage checks + tests pass | Yes (19/19) |
| Pipeline rerunnable from docs | Yes |
