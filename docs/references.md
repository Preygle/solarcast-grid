# References

All entries were checked on 2026-10-26/27 (access date) via the publisher page, DOI resolver, an index (IDEAS/RePEc, OSTI, arXiv, NeurIPS proceedings) or the official documentation. Findings listed are limited to what was verified from the abstract or page; nothing here is paraphrased from memory as if it were read in full.

## Datasets and data portals

| # | Source | Link | Date | Use in project / verified finding |
|---|---|---|---|---|
| D1 | Mukherjee D., Kalita K., Kumar S. *Electricity Demand, Solar and Wind Generation Data (September 2021- June 2025) of India at 1-hour interval*. Mendeley Data, V2. CC BY 4.0. | https://data.mendeley.com/datasets/y58jknpgs8 , doi:10.17632/y58jknpgs8.2 | V2 published 2025-09-15 | **Primary dataset.** Data credited to North-Eastern Regional Load Despatch Centre, Grid-India (Ministry of Power). Our audit (DATA_AUDIT.md) shows the files are actually 5-min / 10-s SCADA for 2021–2023 plus an hourly 2024–2025 report, not uniformly hourly. |
| D2 | NREL. *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*. data.gov / OEDI. CC BY 4.0. | https://catalog.data.gov/dataset/photovoltaic-data-acquisition-pvdaq-public-datasets | first published 2021-12-21; modified 2025-07-31 | Audited, not selected: US site-level PV systems; not Indian grid-level. |
| D3 | NITI Aayog. *India Climate & Energy Dashboard (ICED)*. | https://iced.niti.gov.in/ | accessed 2026-10-26 | Audited, not selected: chart-level XLS downloads; no explicit license; hourly load curve only as a chart export. |
| D4 | Ministry of Power / CEA. *National Power Portal – Published Reports*. | https://npp.gov.in/publishedReports | accessed 2026-10-26 | Audited, not selected: daily/monthly PDF/XLS generation reports; no hourly data, no line-level topology or receiving-end measurements. |
| D5 | Open-Meteo. *Historical Forecast API* (archived NWP model output) and *Historical Weather API*. | https://open-meteo.com/en/docs/historical-forecast-api | accessed 2026-10-27 | Weather-feature experiment (Phase 3). Verified via live API call that hourly shortwave radiation, temperature and cloud cover are returned for 2021-09 onward for Indian coordinates. |

## Duck curve, curtailment and Indian grid context

| # | Source | Link | Date | Verified finding |
|---|---|---|---|---|
| R1 | Denholm P., O'Connell M., Brinkman G., Jorgenson J. *Overgeneration from Solar Energy in California: A Field Guide to the Duck Chart*. NREL/TP-6A20-65023. | https://www.osti.gov/biblio/1233145 | 2015 | Under business-as-usual assumptions, solar near 20% of annual energy could drive marginal curtailment above 30%; distributed PV and storage providing grid services can substantially increase flexibility. The duck chart originates from CAISO's 2013 publication. |
| R2 | Central Electricity Authority. *CEA (Flexible Operation of Coal based Thermal Power Generating Units) Regulations, 2023* (as reported by Powerline and SCC Online). | https://powerline.net.in/2023/03/10/balancing-renewables-cea-sets-flexible-operation-standards-for-tpps/ ; https://scconline.com/blog/?p=282841 | notified Jan 2023 | Minimum power level of 40% for coal units, with 55% as an interim step within one year of notification. Ramp rates 1%/min (40–55%), 2%/min (55–70%), 3%/min (70–100%). Used to parameterise the surplus-risk proxy. Secondary sources only; the gazette text was not retrieved. |
| R3 | Economic Advisory Council to the PM. *The Duck and The Camel: Tracing the Net Load on the Indian Power Grid* (as reported by Energetica India). | https://www.energetica-india.net/news/indias-power-grid-faces-new-challenge-as-solar-growth-shifts-focus-from-capacity-to-flexibility-eac-pm-paper | 2026 | Reports a summer duck-shaped net load and a winter double-peaked shape in India with a growing evening ramp. Context only; the primary paper was not retrieved, so no numbers from it are used. |

## Forecasting

| # | Source | Link | Date | Verified finding |
|---|---|---|---|---|
| F1 | Hong T., Fan S. *Probabilistic electric load forecasting: A tutorial review*. Int. J. Forecasting 32(3):914–938. | doi:10.1016/j.ijforecast.2015.11.011 | 2016 | Reviews probabilistic load forecasting techniques and evaluation; motivates prediction intervals. |
| F2 | Antonanzas J. et al. *Review of photovoltaic power forecasting*. Solar Energy 136:78–111. | doi:10.1016/j.solener.2016.06.069 | 2016 | Reviews PV power forecasting across horizons from seconds to weeks. |
| F3 | Hyndman R.J., Koehler A.B. *Another look at measures of forecast accuracy*. Int. J. Forecasting 22(4):679–688. | doi:10.1016/j.ijforecast.2006.03.001 | 2006 | Shows many percentage measures degenerate (e.g. near-zero actuals) and proposes MASE. Justifies avoiding MAPE for solar and using MAE/RMSE/nMAE/MASE. |
| F4 | Ke G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 30. | https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf | 2017 | GOSS + EFB give large speedups at similar accuracy to conventional GBDT. |
| F5 | Hochreiter S., Schmidhuber J. *Long Short-Term Memory*. Neural Computation 9(8):1735–1780. | doi:10.1162/neco.1997.9.8.1735 | 1997 | Introduces LSTM for learning long time lags with gradient-based training. |

## Grid modelling, graph analysis and power flow

| # | Source | Link | Date | Verified finding |
|---|---|---|---|---|
| G1 | Thurner L. et al. *pandapower — An Open-Source Python Tool for Convenient Modeling, Analysis, and Optimization of Electric Power Systems*. IEEE Trans. Power Syst. 33(6):6510–6521. | doi:10.1109/TPWRS.2018.2829021 ; arXiv:1709.06743 | 2018 | BSD-licensed Python tool providing power flow, OPF, state estimation, topological graph searches and IEC 60909 short circuit. |
| G2 | Baran M.E., Wu F.F. *Network reconfiguration in distribution systems for loss reduction and load balancing*. IEEE Trans. Power Delivery 4(2):1401–1407. | (cited via later works, e.g. https://arxiv.org/pdf/2410.04604) | 1989 | Origin of the 33-bus test feeder and branch-exchange reconfiguration for loss reduction. The original paper was not retrieved; the network data come from pandapower's `case33bw` (MATPOWER-derived). |
| G3 | pandapower documentation, *Power System Test Cases — case33bw*. | https://pandapower.readthedocs.io/en/stable/networks/power_system_test_cases.html | accessed 2026-10-26 | `case33bw()` loads the Baran & Wu feeder from a MATPOWER-provided file. |
| G4 | Pagani G.A., Aiello M. *The Power Grid as a complex network: A survey*. Physica A 392(11):2688–2700. | doi:10.1016/j.physa.2013.01.023 ; arXiv:1105.3338 | 2013 | Surveys complex-network (graph) analysis of power grids: topological and physical properties, graph indicators and reliability. Supports the graph-analysis component but also its limitation: pure topology does not determine electrical flows. |

## Software (versions installed and verified in this environment)

Python 3.13.3, numpy 2.3.3, pandas 2.3.3, scikit-learn 1.6.1, xgboost 3.0.5, lightgbm 4.7.0, torch 2.10.0+cpu, networkx 3.4.2, pandapower 3.5.6, pulp 4.2.0 with HiGHS (highspy 1.15.1), streamlit 1.45.1, plotly 6.9.0, matplotlib 3.10.3, statsmodels 0.15.0, python-calamine (xlsx reader).
