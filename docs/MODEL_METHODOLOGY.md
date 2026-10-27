# Forecasting Methodology

## Tasks
| Task | Target (MW, hourly mean, IST) | Issue time τ | Leads |
|---|---|---|---|
| `solar_h1`, `demand_h1`, `wind_h1`, `net_load_h1` | value of hour t | t − 1 h | 1 h |
| `solar_da`, `demand_da`, `wind_da`, `net_load_da` | all 24 hours of day D+1 | 10:00 on D (data through 09:00–10:00) | 15–38 h |

Net load is forecast directly (`net_load_*`). It is also composed from the component forecasts (demand − solar − wind) in the optimisation interface (`Forecaster.day_ahead`).

## Feature/label alignment
Each row is a target hour t. Features use only timestamps ≤ τ(t):
- Target lags: h1 {1, 2, 3, 24, 48, 168} h; da {48, 72, 168} h. Every day-ahead lag must be ≥ 38 h, the longest lead, so that no target hour ever sees data after 09:00 on D. This is enforced by `check_no_leakage`.
- Lags of the other series: h1 {1, 24}; da {48, 168}.
- Statistics at τ: last value; 24 h mean/std/max; 168 h mean; 720 h max (capacity/level proxy); mean of the same clock-hour over the 7 most recent available days.
- Calendar of the target hour: hour (raw + sin/cos), day-of-week, weekend flag, day-of-year harmonics, month.
- Tested by perturbation: values after τ are multiplied by 1.37, and the row's features must not change (`tests/test_features_baselines.py`). The inference path is tested the same way (`tests/test_models_inference.py`).

## Splits and protocol
Train 2021-09-01 → 2024-06-30 · Validation 2024-07-01 → 2024-12-31 · Test 2025-01-01 → 2025-06-24.
1. For each family, bounded random search with a fixed seed (Ridge 4, RF 3, HistGB 6, LightGBM 10, XGBoost 6 configurations) × target transform {none, level-ratio}. Models are fitted on train and scored by validation MAE. LightGBM and XGBoost use early stopping on validation (150 rounds).
2. The best configuration per family is refitted on train + validation, reusing the early-stopped iteration count, and scored **once** on test.
3. Per task, the selected model is the family (baselines included) with the lowest **validation** MAE. Test scores are never used for selection. The best model on test is reported separately for transparency.
4. LSTM (PyTorch, CPU): input = 168-h sequence ending at τ of z-scored demand/solar/wind + missing flag + hour harmonics; static = target-hour calendar + lead. Level-ratio target, L1 loss, Adam 1e-3, batch 256, ≤ 25 epochs, early stopping (patience 4) on validation. Day-ahead tasks try hidden ∈ {32, 64}; hourly tasks use hidden 64 only, to bound CPU time. The model is refitted on train + validation for the best epoch count. Seed 42.
5. Post-processing: solar and wind are clipped at ≥ 0. Solar is set to 0 in hours 20:00–03:59, which matches the physical definition used in cleaning.

## Why a level-ratio target (key decision)
- **Problem.** All-India solar capacity grows month by month. Test-period peaks (≈64 GW) exceed every value seen in training (≤ 55 GW). Tree ensembles cannot predict outside the training range. In a pilot run, LightGBM's day-ahead solar MAE was 1,178 MW on validation but 2,621 MW on test, worse than the seasonal-naive baseline.
- **Alternatives.** Linear models, which extrapolate but are less flexible; differencing; capacity-normalised targets.
- **Choice.** `LevelRatioRegressor`. The target and the target's own lag/rolling features are divided by a level known at τ (rolling 30-day max for solar and wind, 168-h mean for demand and net load). The level column itself is dropped, and predictions are multiplied back. The model therefore learns shapes rather than magnitudes.
- **Evidence.** In the same pilot, LightGBM day-ahead solar test MAE fell to 1,266 MW. Whether to use the transform is still chosen per family on validation.
- **Assumption.** The shape relative to recent level is stable. It can fail at abrupt level changes.

## Prediction intervals
Split-conformal by hour of day. Absolute validation residuals of the train-only fit of the selected model give the 80th percentile per hour, which is applied symmetrically to test predictions (lower bound clipped at 0 for solar and wind). Empirical test coverage is reported. Under distribution shift it can fall below the nominal 80%.

## Metrics
MAE, RMSE, bias (MW), R², nMAE = MAE / mean |actual|, MASE (scale = in-sample MAE of the 24-h seasonal-naive forecast on train). MAPE is reported only for demand and net load. It is not used for solar or wind, which reach zero (Hyndman & Koehler 2006). Solar additionally gets daylight-only MAE.

## Interpretation and robustness (`scripts/evaluate_models.py`)
Permutation importance on test (MAE increase). Breakdowns by hour of day, lead time, high-solar hours (top 10%), evening hours 16–20, steep-ramp hours (top 10% of 3-h net-load increase), weekends and month. Ablation: full vs no-lags vs calendar-only, plus a separately labelled weather experiment.
