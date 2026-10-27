# Model Evaluation (FORECAST vs OBSERVED)

Test period 2025-01-01 → 2025-06-24 (≈4,150 hours per task), evaluated once after selection on validation (2024-07 → 2024-12). Sources: `results/model_comparison.csv`, `results/metrics.json`, `results/forecast_regime_breakdown.csv`, `results/feature_ablation_weather.csv`, `results/feature_importance.csv`, figures `results/figures/fc_*.png`.

## 1. Consolidated comparison: MAE in MW (validation / test), sorted by validation

| Model | solar da | solar h1 | demand da | demand h1 | wind da | wind h1 | net load da | net load h1 |
|---|---|---|---|---|---|---|---|---|
| Ridge | 1,226 / 1,378 | 649 / 851 | 4,443 / 4,905 | 1,598 / 1,953 | **1,980 / 2,040** | 358 / 416 | 5,988 / 6,199 | 1,941 / 2,387 |
| Random Forest | 1,123 / 1,238 | 532 / 597 | 4,249 / 4,619 | 1,337 / 1,707 | 2,086 / 2,199 | 362 / 410 | 5,756 / 6,284 | 1,908 / 2,506 |
| HistGradientBoosting | **1,081 / 1,277** | 487 / 580 | 4,275 / 4,513 | 1,257 / 1,461 | 2,037 / 2,341 | 346 / 392 | 5,713 / 6,343 | 1,616 / 2,035 |
| LightGBM | 1,085 / 1,266 | **477 / 559** | 4,229 / 4,337 | **1,234 / 1,427** | 1,995 / 2,237 | **341 / 389** | 5,602 / 6,245 | **1,600 / 2,016** |
| XGBoost | 1,092 / 1,248 | 482 / 586 | **4,226 / 4,631** | 1,252 / 1,497 | 2,024 / 2,240 | 344 / 389 | **5,585 / 6,488** | 1,616 / 2,023 |
| LSTM | 1,314 / 1,937 | 1,109 / 1,664 | 6,108 / 6,683 | 4,654 / 5,580 | 2,161 / 2,033 | 876 / 1,133 | 7,839 / 8,972 | 3,639 / 3,893 |
| Best naive baseline | 1,350 / 1,295 | 1,040 / 1,055 | 6,554 / 7,003* | 3,869 / 4,396 | 2,682 / 2,705 | 481 / 612 | 7,753 / 7,781* | 5,287 / 6,235* |

**Bold** = selected (lowest validation MAE). \* For these tasks the baseline with the lowest validation MAE and the one with the lowest test MAE differ; the test figure shown is the best baseline on test, which is the stricter comparison.

## 2. Selected models (test)

| Task | Selected | MAE (MW) | RMSE (MW) | R² | nMAE % | Best baseline (test MAE) | Improvement | 80% PI coverage |
|---|---|---|---|---|---|---|---|---|
| solar_da | HistGB | 1,277 | 2,477 | 0.988 | 6.93 | seasonal-naive 1,295 | **1.4%** | 81.5% |
| solar_h1 | LightGBM | 559 | 1,119 | 0.998 | 3.04 | seasonal-naive 1,055 | 47.0% | 80.6% |
| demand_da | XGBoost | 4,631 | 6,492 | 0.887 | 2.33 | seasonal-naive 7,003 | 33.9% | 75.1% |
| demand_h1 | LightGBM | 1,427 | 2,032 | 0.989 | 0.72 | persistence 4,396 | 67.6% | 73.2% |
| wind_da | Ridge | 2,040 | 2,716 | 0.783 | 22.0 | same-hour 7-day mean 2,694 | 24.3% | 77.9% |
| wind_h1 | LightGBM | 389 | 535 | 0.992 | 4.20 | persistence 612 | 36.5% | 75.1% |
| net_load_da | XGBoost | 6,488 | 8,517 | 0.830 | 3.81 | seasonal-naive 7,781 | 16.6% | 73.3% |
| net_load_h1 | LightGBM | 2,016 | 2,773 | 0.982 | 1.18 | seasonal-naive 6,235 | 67.7% | 68.5% |

## 3. Findings

1. **Gradient-boosted trees win 6 of 8 tasks on validation; Ridge wins day-ahead wind.** Differences between LightGBM, XGBoost and HistGB are small: within ≈3% of each other on validation and ≈7% on test for every task. The choice among them is close to noise.
2. **The LSTM did not justify its complexity.** It was never selected and was worst among the ML models on 7 of 8 tasks. It was best on test only for day-ahead wind (2,033 vs 2,040 MW for Ridge), a negligible margin, and it lost on validation. Contributing factors: one compact architecture, ≤ 25 epochs, and a single configuration for hourly tasks to bound CPU time. A larger search might narrow the gap, but there is no evidence here that it would overturn the ranking.
3. **Day-ahead solar is hard to beat without weather.** The selected model improves on the seasonal-naive baseline by only 1.4% overall. It is **worse** than the baseline in the top-10% solar hours (2,442 vs 2,167 MW, −12.7%) and on the steepest-ramp days (−21.5%), and worse in February and March 2025. Clear-sky spring days are close to persistent, and the ML model smooths peaks. 1-hour-ahead solar improves by 47%.
4. **Weather matters for solar** (labelled experiment, LightGBM day-ahead). Archived NWP features reduce solar test MAE from 1,266 to 1,134 MW (−10.5%), demand from 4,337 to 4,213 (−2.8%) and net load from 6,245 to 6,123 (−2.0%). The NWP archive is stitched from short-lead runs, so these gains are an **upper bound** for a real 10:00 D−1 forecast.
5. **Ablation.** Without explicit lag features the error barely changes (solar 1,278, demand 4,416, net load 6,298), because the same-clock-hour 7-day mean and the issue-time statistics carry the information. Calendar-only models are 3–5× worse (solar 5,887, demand 14,494, net load 12,611). Recent history is essential; its exact encoding is not.
6. **Feature importance** (permutation on test, MAE increase). For day-ahead tasks, the same-hour 7-day mean dominates (solar +16,578 MW, demand +7,056, net load +6,217), followed by the 48-h lag and the last value at issue time.
7. **Distribution shift.** Test errors exceed validation errors for most tasks (e.g. net_load_h1 1,600 → 2,016 MW). Conformal 80% intervals under-cover on 6 of 8 tasks (68–78%). The 2025 period has higher solar and demand levels than training. Without the level-ratio transform, tree models were ~2× worse on day-ahead solar test data (pilot: 2,621 vs 1,266 MW).
8. **Horizon.** 1-hour-ahead errors are 2–5× smaller than day-ahead (demand nMAE 0.72% vs 2.33%; net load 1.18% vs 3.81%). Day-ahead error by lead hour peaks around midday for solar and at the evening peak for demand (`fc_error_by_lead.png`, `fc_mae_by_hour.png`).
9. **Evening ramp hours (16–20 h).** Day-ahead net load beats the baseline by 24.6% (6,410 vs 8,501 MW), more than its overall 16.6% gain. Hour-ahead net load reaches 74.4%.

## 4. What is used downstream
The optimisation phase uses the **selected day-ahead solar (HistGB) and demand (XGBoost)** models through `src/inference/forecast.py`. The models are reloaded from `models/`, and features are rebuilt from history truncated at the issue time. A test checks that data after the issue time cannot change the forecast.
