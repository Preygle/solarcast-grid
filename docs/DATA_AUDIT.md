# Data Audit

Audit date: 2026-10-27. Numbers below come from `scripts/audit_raw.py` (output: `results/eda/raw_audit.csv`) run on the downloaded files, all verified against the SHA-256 hashes published by Mendeley (`scripts/download_data.py`).

## 1. Selected primary dataset — Mendeley y58jknpgs8 v2 (Grid-India SCADA)

| Item | Finding |
|---|---|
| Source | Mendeley Data, doi:10.17632/y58jknpgs8.2, contributors D. Mukherjee, K. Kalita (IIT Guwahati), S. Kumar (POWERGRID). Data credited to NERLDC, Grid-India. |
| License | CC BY 4.0 (attribution required; redistribution allowed). Raw files are kept out of git only because of size (748 MB); `scripts/download_data.py` re-fetches them. |
| Access | Public REST API `public-api/datasets/y58jknpgs8/files` → 29 xlsx files with download URLs and SHA-256 hashes. No login needed. |
| Geography | All-India aggregate (sum of the five regional grids, NLDC points). **No state, bus or line resolution.** |
| Timezone | Not stated. Assumed IST (UTC+5:30); India has no DST. To be checked in Phase 2 from the solar diurnal peak (should be near 12:00–13:00 IST). |

### 1.1 File layouts (three distinct formats)

| Period | Files | Resolution | Rows | Columns (SCADA point → our name, MW) |
|---|---|---|---|---|
| 2021-09 → 2022-09 | 13 monthly | **5 min** | 8,064–8,928 / month | `TOTAL_THM_ONLY|P` thermal (coal+lignite), `TOTAL_NLDC_HYD|P` hydro, `TOTAL_NLDC_GAS|P` gas, `TOTAL_NLDC_NPC|P` nuclear, `ALL_INDIA_WIND|P` wind, `ALL_IND_SOLAR|P` solar, `NLDC_DEMAND|P` demand met; plus derived spreadsheet columns (Solar = solar clipped at 0, Solar+Wind, Total) and inertia points (ignored) |
| 2022-10 → 2023-12 | 15 monthly | **10 s** | 241,920–267,840 / month | same points |
| 2024-01-01 → 2025-06-24 | 1 combined | **1 h** | 12,984 (= 541 days × 24, complete) | Demand, Wind, Solar, Total Generation only. **No thermal/hydro/gas/nuclear split.** Three trailing summary rows (max timestamp / average / sum) must be dropped. |

The dataset title says "1-hour interval", but only the 2024–25 file is hourly. The description on the dataset page says the data ends December 2023; the title and the files show it extends to 2025-06-24.

### 1.2 Quality findings

| Check | Result |
|---|---|
| Duplicate timestamps | 0 in every file |
| Non-monotonic / gaps | 0; every file has a single constant step (max gap = nominal step) |
| Explicit NaN | 0 in all value columns |
| **Telemetry dropouts encoded as 0** | Minimum demand = 0 MW in 25 of 28 monthly files; thermal, hydro and nuclear also hit exactly 0. All-India demand of 0 MW is physically impossible, so these are dropouts and must be treated as missing. |
| **Spikes** | Demand max 402,065 MW (Jan 2023) and 466,674 MW (Sep 2023), against a true all-India peak of ~240 GW; nuclear 38,397 MW (Sep 2021) against ~7 GW installed. These need a robust outlier filter. |
| Negative solar | Night-time solar is slightly negative (−20 to −450 MW typical; −1,887 MW in Oct 2023) from metering offsets. Will be clipped at 0 and the clipping documented. |
| Negative gas | Gas reaches −5,375 MW (May 2023), likely a telemetry error. Flag it. |
| Level growth | Solar maximum rises from ~37 GW (2021) to ~44 GW (2023) to 64.7 GW (2024–25), which matches capacity additions. Non-stationarity has to be handled in modelling (e.g. capacity-normalised targets or recency weighting). |
| Format break at 2024-01 | Source changes from a SCADA point export to an hourly "Report". The hourly aggregation method is undocumented (snapshot or hourly mean). Phase 2 compares level and shape at the boundary and records the result. |

### 1.3 Suitability for prediction targets

| Target | Supported? | Notes |
|---|---|---|
| Solar generation forecast | **Yes** (2021-09 → 2025-06) | All-India aggregate MW |
| Demand forecast | **Yes** | "Demand met" (NLDC), not unrestricted demand; load shedding is not observable |
| Net load (demand − solar − wind) | **Yes** | Derived |
| Surplus-solar risk | **Proxy only** | Thermal/hydro/nuclear/gas split exists only to 2023-12. No unit commitment, technical-minimum, reserve, import/export or curtailment records exist, so an **actual** surplus cannot be computed. We define a scenario-based proxy (PROJECT_SPEC §4.4). |
| Technical losses / receiving-end power | **No** | No line-level topology, impedances or receiving-end metering. These come from a power-flow benchmark (Section 3). |
| Weather inputs | **Not included** | Taken from Open-Meteo archived NWP forecasts (Section 2). |

## 2. Supporting dataset — Open-Meteo weather (Phase 3 experiment)

Verified by a live API call: the Historical Forecast API returns hourly `shortwave_radiation`, `temperature_2m` and `cloud_cover` from 2021-09-01 for Indian coordinates, and the archive (reanalysis) API covers 2025. Planned use: radiation/cloud at major solar hubs and temperature at large load centres. **Caveat:** the historical forecast series is stitched from the shortest-lead part of successive model runs, so for day-ahead use it is optimistic. It is reported as a separate, labelled experiment and never mixed into the baseline feature set.

## 3. Grid network data — benchmark, not India

No public source audited here provides Indian line-level topology, impedances or receiving-end measurements. We therefore use the **IEEE/Baran & Wu 33-bus 12.66 kV distribution feeder** (`pandapower.networks.case33bw`, MATPOWER-derived). Verified in this environment:

- 33 buses, 32 loads (3.715 MW, 2.300 Mvar total), 1 external grid (slack at bus 0), 37 lines of which 5 are normally-open tie lines.
- Base-case AC power flow converges with **total losses 0.2027 MW** and **minimum voltage 0.9131 pu**, which matches the 202.7 kW / 0.913 pu values widely reported for this feeder.
- Bus voltage limits in the case file: 0.9–1.1 pu.
- **Line thermal ratings are absent** (`max_i_ka = 99999`). A documented assumed rating will be added in Phase 4 and labelled as an assumption.

All Problem B/C results are **simulated**. They are not Indian grid measurements.

## 4. Audited but not selected

| Source | Reason not selected |
|---|---|
| NREL PVDAQ (CC BY 4.0) | US site-level PV systems; does not represent Indian grid-level solar or demand. |
| ICED (NITI Aayog) | Data only as chart exports, with no explicit license; the hourly load curve is not available as a bulk time series; some charts show "Source: undefined". |
| National Power Portal | Daily/monthly PDF/XLS reports. No hourly series and no line-level topology or receiving-end measurements. |
