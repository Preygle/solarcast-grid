# SolarCast-Grid

Predictive analysis and graph-based optimisation for solar-energy utilisation and power delivery.

- **Forecasting (primary contribution):** all-India solar, demand, wind and net load at 1-hour-ahead and day-ahead horizons. Built from real Grid-India SCADA data (Sep 2021 – Jun 2025) and compared across baselines, Ridge, Random Forest, HistGradientBoosting, LightGBM, XGBoost and an LSTM.
- **Duck curve and surplus risk:** net-load shape, evening ramps, and an assumption-based surplus-solar-risk proxy.
- **Grid:** the IEEE/Baran-Wu 33-bus feeder as a graph (NetworkX) with AC power flow (pandapower). It predicts technical losses and receiving-end power on held-out scenarios.
- **Optimisation:** a forecast-informed 24-hour LP (battery + flexible load + curtailment, LinDistFlow constraints, HiGHS). Every schedule is validated by AC power flow and compared with baselines.
- **Dashboard:** Streamlit + Plotly, six pages; it reads saved artefacts only.

> Data classes are labelled throughout as OBSERVED, DERIVED, FORECAST, SIMULATED or OPTIMISED. The grid is a **benchmark feeder, not India's grid**, and all grid results are SIMULATED.

## Quick start (from the project root)

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py        # 748 MB raw data (Mendeley y58jknpgs8 v2, CC BY 4.0), SHA-256 verified
python scripts/build_dataset.py        # -> data/processed/hourly.parquet (already committed)
python -m streamlit run app/app.py     # dashboard (uses committed artefacts; no retraining)
python -m pytest -q                    # tests
```

The processed hourly table, trained models and results are committed, so the dashboard and tests run without downloading the raw data.

## Full reproduction

```bash
python scripts/audit_raw.py            # raw-file audit            -> results/eda/raw_audit.csv
python scripts/build_dataset.py        # cleaning + hourly table   -> data/processed/, results/eda/preprocessing_log.json
python scripts/run_eda.py              # EDA, duck curve, surplus proxy -> results/eda/
python scripts/run_baselines.py        # naive baselines           -> results/baseline_metrics.csv
python -c "import sys; sys.path.insert(0,'.'); from src.data.weather import build_weather; build_weather()"   # weather experiment inputs
python scripts/train_models.py         # all forecasting models (~1 h CPU) -> models/, results/model_comparison.csv
python scripts/evaluate_models.py      # figures, breakdowns, importance, ablation/weather -> results/
python scripts/run_grid_analysis.py    # graph metrics, reconfiguration, battery siting -> results/grid_scenarios/
python scripts/run_receiving_power.py  # 12k AC power flows + loss/receiving-end models -> results/grid_model_comparison.csv
python scripts/run_optimization.py     # baseline vs forecast-informed LP, AC-validated -> results/optimization_comparison.csv
```

## Project map

| Path | Content |
|---|---|
| `src/data` | raw loaders, cleaning pipeline, weather fetch |
| `src/features` | leak-safe feature builder (issue-time aware), chronological splits |
| `src/analysis` | EDA figures, duck-curve metrics, surplus proxy |
| `src/models` | baselines, tabular models (level-ratio wrapper), LSTM |
| `src/training`, `src/evaluation`, `src/inference` | training/tuning protocol, metrics, prediction interface |
| `src/grid`, `src/power_flow`, `src/simulation` | feeder model, graph analysis, AC power flow runner, scenario generator |
| `src/receiving_power`, `src/optimization` | loss / receiving-end models; LP and evaluation |
| `app/` | Streamlit dashboard |
| `docs/` | specification, data audit, pipeline, methodology, phase reports, final results |
| `results/`, `models/` | generated artefacts |

## Documentation
`docs/PROJECT_SPEC.md` · `docs/DATA_AUDIT.md` · `docs/DATA_PIPELINE.md` · `docs/MODEL_METHODOLOGY.md` · `docs/MODEL_EVALUATION.md` · `docs/GRID_MODEL.md` · `docs/OPTIMIZATION_METHODOLOGY.md` · `docs/METHODOLOGY_AND_LIMITATIONS.md` · `docs/FINAL_RESULTS.md` · `docs/USER_GUIDE.md` · `docs/FINAL_REPORT.md` · `docs/references.md` · `docs/PHASE_*_REPORT.md`

## Data attribution
Mukherjee D., Kalita K., Kumar S. (2025). *Electricity Demand, Solar and Wind Generation Data (September 2021- June 2025) of India at 1-hour interval*. Mendeley Data, V2. doi:10.17632/y58jknpgs8.2. Licensed CC BY 4.0. Data originate from NERLDC, Grid-India. The derived file `data/processed/hourly.parquet` is a cleaned, hourly-resampled version (see `docs/DATA_PIPELINE.md`).
Weather (experiment only): Open-Meteo Historical Forecast API.
Benchmark grid: Baran & Wu (1989) 33-bus feeder via pandapower `case33bw`.
