# User Guide

## 1. Environment
Tested with Windows 11, Python 3.13.3 and CPU only. Install the dependencies from the project root:
```bash
python -m pip install -r requirements.txt
```
`pulp` is pinned to 3.3.2. PuLP 4.x replaced the classic `LpVariable` API used in `src/optimization/lp.py`. HiGHS is used through `highspy`.

## 2. Launch the dashboard
```bash
python -m streamlit run app/app.py
```
Open the URL Streamlit prints (default http://localhost:8501). The dashboard loads saved artefacts only and never retrains. If an artefact is missing, the page names the command that creates it.

| Page | What you can do |
|---|---|
| Overview | purpose, workflow, data provenance, model status, limitations |
| Forecasting | choose target, horizon and models; zoom the test period; view validation/test metrics; run a **live day-ahead forecast** from the saved models for any test day |
| Duck curve | seasonal net-load profiles by year, ramp statistics, actual-vs-forecast net load, surplus-risk proxy and its definition |
| Optimisation | saved baseline-vs-optimised results; **run the LP for a chosen day** with your battery size, power, initial SoC, flex share, PV multiplier and export limit; every result is AC-validated |
| Grid & power delivery | feeder graph coloured by AC voltage; choose configuration, load, PV and battery; compare **AC-simulated vs model-predicted** received power, losses and loss % |
| Model evaluation | consolidated comparison, error breakdowns, permutation importance, ablation/weather experiment, loss-model evaluation and fresh power-flow check |

## 3. Reproduce results
See README "Full reproduction". Approximate CPU times on a 16-thread laptop:
- dataset build ~2 min
- EDA <1 min
- model training ~45–60 min
- evaluation ~15 min
- grid analysis ~3 min
- receiving-end scenarios + models ~10 min
- optimisation over all test days ~25–40 min

## 4. Tests
```bash
python -m pytest -q
```
Tests that need generated artefacts (processed data, trained models) are skipped automatically if those are absent.

## 5. Interpreting labels
**OBSERVED** = Grid-India data after documented cleaning. **DERIVED** = arithmetic on observations. **FORECAST** = model output. **SIMULATED** = pandapower AC power flow on the benchmark feeder. **OPTIMISED** = LP decisions. Feeder MW values are not Indian measurements.
