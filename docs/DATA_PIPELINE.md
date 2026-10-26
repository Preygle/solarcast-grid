# Data Pipeline

## Reproduce

```bash
python scripts/download_data.py      # 29 xlsx files, 748 MB, SHA-256 verified (Mendeley y58jknpgs8 v2, CC BY 4.0)
python scripts/audit_raw.py          # results/eda/raw_audit.csv
python scripts/build_dataset.py      # data/processed/hourly.parquet + results/eda/preprocessing_log.json (~2 min first run)
python scripts/run_eda.py            # EDA figures, duck-curve and surplus-proxy tables in results/eda/
python scripts/run_baselines.py      # results/baseline_metrics.csv
python -m pytest -q
```
`build_dataset.py` caches per-month hourly frames in `data/interim/` (git-ignored). Use `--no-cache` to re-parse the raw files.

## Cleaning steps (src/data/preprocess.py)

| # | Step | Resolution | Rule | Count (all files) |
|---|---|---|---|---|
| 1 | Dropout rows | raw | demand == 0 → whole row NaN | 1,080 samples |
| 1b | Zero dropouts | raw | exact 0 in non-solar columns → NaN | hydro 8,641; nuclear 3 |
| 2 | Physical bounds | raw | outside `BOUNDS` → NaN | demand 2, wind 33, hydro 1, gas 2,234 (negative), nuclear 304 |
| 3 | Hampel spikes | raw (1 h window) | \|x − median\| > max(6·1.4826·MAD, floor) | demand 4,307; solar 3,389; wind 1,761; thermal 1,804; hydro 852; gas 2,624; nuclear 1,290 |
| 4 | Negative solar | raw | clipped to 0 | 306,403 samples (night metering offset) |
| 5 | Hourly mean | 1 h | mean over [t, t+1h); NaN if < 50% samples valid | — |
| 6 | Hour-level spikes | 1 h (5 h window) | Hampel floors: demand 12 GW, thermal 12 GW, wind 6 GW | demand 15, thermal 9, wind 2 h |
| 6b | Sustained demand dropouts | 1 h | same-clock-hour Hampel over 15 days, floor 25 GW | 44 h |
| 7 | Frozen telemetry | 1 h | ≥ 3 identical consecutive non-zero values → NaN | demand 20, solar 80, wind 282 h |
| 8 | Solar stuck values | 1 h | same-clock-hour Hampel over 15 days, floor 1.5 GW | 277 h |
| 9 | Solar night offset | 1 h | subtract the day's median 00–03 h value (mean 89 MW, max 2,028 MW); hours 20–03 set to 0 | 11,144 night hours set to 0 |
| 10 | Short gaps | 1 h | linear interpolation only for gaps ≤ 2 h, flagged `<col>_imputed` | demand 440, solar 475, wind 27 h |

Remaining NaN after cleaning (of 33,432 hours): demand 65, solar 143, wind 295. Thermal/hydro/gas/nuclear: 13–39 h before 2024; absent from 2024 on by source design. **No values are fabricated beyond the ≤ 2 h interpolation and the physical night-time zero for solar.**

Every rule is a deliberate trade-off. The 25 GW floor in step 6b sits above typical holiday demand reductions, so genuine low-demand days are kept. Some residual anomalies may remain undetected, e.g. evening solar values that are stuck but below 1.5 GW.

## Timestamp and unit conventions
- Timezone: **IST**, no DST. Check: mean solar output peaks in the 12:00–13:00 hour in every year 2021–2025.
- Hourly label = start of hour. Values = mean MW over the hour, so MWh per hour = MW value.
- All values are all-India aggregates in MW. Unit check: median demand ≈ 1.8 × 10⁵ MW (tested).

## Data dictionary — `data/processed/hourly.parquet`

| Column | Unit | Class | Definition | Coverage |
|---|---|---|---|---|
| `timestamp` (index) | IST | — | start of hour | 2021-09-01 00:00 → 2025-06-24 23:00 |
| `demand_mw` | MW | OBSERVED | NLDC demand met | all |
| `solar_mw` | MW | OBSERVED (cleaned) | all-India solar, offset-corrected, ≥ 0 | all |
| `wind_mw` | MW | OBSERVED | all-India wind | all |
| `thermal_mw` | MW | OBSERVED | coal + lignite (TOTAL_THM_ONLY) | ≤ 2023-12 |
| `hydro_mw`, `gas_mw`, `nuclear_mw` | MW | OBSERVED | NLDC totals | ≤ 2023-12 |
| `total_generation_mw` | MW | OBSERVED (raw) | total generation from the 2024–25 report; not cleaned, not used in models | 2024-01 → |
| `net_load_mw` | MW | DERIVED | demand − solar − wind | all |
| `<col>_imputed` | bool | flag | value interpolated (gap ≤ 2 h) | — |
| `source_file` | str | provenance | raw workbook name | — |

## Format break at 2024-01-01
The source changes from SCADA point exports (mean of 5-min/10-s samples) to an hourly report whose aggregation method is undocumented. Checked: hour-to-hour mean |Δdemand| is 4.6 GW (Oct–Dec 2023) vs 5.0 GW (Jan–Mar 2024), and second differences are 3.0 vs 3.2 GW. The roughness is similar, so the 2024 values are probably hourly means or close to them. The difference cannot be fully resolved. Models use a capacity/level proxy and recent-window features so that a small level shift does not dominate.

## Features (src/features/build.py)
- **Issue time τ.** `h1`: τ = t − 1 h. `da`: τ = 09:00 on day D (forecast issued at 10:00 D for all hours of D+1).
- Target lags: h1 {1, 2, 3, 24, 48, 168}; da {48, 72, 168}. The guard `check_no_leakage` requires lag ≥ 38 h for da (largest lead: 23:00 D+1 − 09:00 D).
- Exogenous lags of the other series (h1: 1, 24; da: 48, 168).
- Statistics at τ: last value, 24 h mean/std/max, 168 h mean, 720 h max (capacity/level proxy), same-clock-hour mean of the previous 7 available days.
- Calendar: hour (raw + sin/cos), day-of-week, weekend flag, day-of-year sin/cos, month. **Holidays are not included**: no reliable machine-readable all-India holiday calendar was used, and holidays vary by state.
- Weather: optional, separate experiment (Phase 3).
- **Leakage test:** for sampled rows, multiplying every value after τ by 1.37 leaves that row's features unchanged (`tests/test_features_baselines.py`).

## Splits
Train 2021-09-01 → 2024-06-30 · Validation 2024-07-01 → 2024-12-31 · Test 2025-01-01 → 2025-06-24. Chronological, disjoint, never shuffled (tested).
