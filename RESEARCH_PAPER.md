# Forecasting Unaccompanied Children in HHS Care: Exploratory Evidence, Predictive Performance, and Capacity-Planning Recommendations

## Abstract

This study examines operational data on unaccompanied children (UAC) moving through U.S. Customs and Border Protection (CBP) and Department of Health and Human Services (HHS) care. The objectives are to describe changes in demand and care occupancy, identify operational relationships, and assess whether short-term forecasts can support capacity planning. The dataset contains 720 reported observations from 12 January 2023 to 21 December 2025. It includes CBP apprehensions, children in CBP custody, transfers from CBP, children in HHS care, and discharges from HHS care. Exploratory analysis shows a highly non-stationary system: HHS care reached 11,516 children on 20 December 2023, fell to 1,972 on 21 August 2025, and ended at 2,484. Annual mean HHS care declined from 8,646 in 2023 to 2,543 in 2025. Transfers and discharges were strongly associated with HHS occupancy, but these contemporaneous correlations cannot establish causality. Chronological holdout and walk-forward tests show that forecast accuracy deteriorates substantially as the horizon increases. Simple exponential-smoothing and seasonal statistical models generally provide more reliable evidence than the tested machine-learning models. The results support a short-horizon, scenario-based forecasting process with frequent recalibration, explicit reporting-gap treatment, and operational thresholds tied to forecast intervals rather than single point estimates.

**Keywords:** unaccompanied children, HHS, ORR, exploratory data analysis, time-series forecasting, capacity planning, SARIMA, exponential smoothing

## 1. Introduction

The Office of Refugee Resettlement (ORR), within HHS, receives unaccompanied children referred by other federal agencies, usually the Department of Homeland Security. HHS provides care while children await safe placement, commonly with a vetted sponsor ([HHS, “Unaccompanied Children Information”](https://www.hhs.gov/programs/social-services/unaccompanied-children/index.html); [HHS FY 2025 Congressional Justification](https://www.acf.hhs.gov/sites/default/files/documents/olab/fy-2025-congressional-justification.pdf)). Because arrivals, transfers, and releases can change quickly, care providers must make decisions about beds, staffing, transport, health services, education, and sponsor-processing capacity under uncertainty.

This paper addresses three questions:

1. How did CBP and HHS operational measures change between January 2023 and December 2025?
2. Which observed measures are most closely associated with HHS care occupancy and net flow?
3. Which forecasting approaches offer the strongest evidence for short-term operational planning?

The paper contributes an auditable exploratory analysis and a conservative interpretation of predictive results. It does not attempt to identify the causes of migration or estimate the effect of policy changes.

## 2. Data and Methods

### 2.1 Data

The local source file contains five reported measures:

- children apprehended and placed in CBP custody;
- children in CBP custody;
- children transferred out of CBP custody;
- children in HHS care; and
- children discharged from HHS care.

After parsing dates and numeric fields, the pipeline produced 720 unique, complete observations covering 1,075 calendar days. There were no missing values, duplicate reporting dates, or negative counts in the processed data. Only 67.0% of calendar dates were reported (720/1,075); 355 dates were unreported. The median interval between reports was one day, the mean was 1.49 days, and the maximum was ten days. Unreported dates were flagged and were not interpreted as days with zero activity.

`net_hhs_flow` was constructed as:

\[
\text{Net HHS flow}_t = \text{CBP transfers to HHS}_t - \text{HHS discharges}_t.
\]

A positive value therefore indicates more reported transfers than discharges, while a negative value indicates the reverse. This constructed measure is an operational proxy; it is not guaranteed to reconcile exactly with the change in HHS occupancy because definitions, timing, other inflows/outflows, and reporting intervals may differ.

### 2.2 Exploratory data analysis

The analysis used descriptive statistics, annual summaries, extrema, report-to-report changes, Pearson correlations, and an additive decomposition of a daily interpolated HHS-care series. Interpolation was used only for decomposition and visualization—not to relabel unreported dates as observed values. Correlations are descriptive, contemporaneous, and vulnerable to shared trends and serial dependence.

### 2.3 Forecasting and validation

Forecast targets were HHS care, HHS discharges, and net HHS flow at horizons of 1, 7, and 14 future **reported observations**. These are not necessarily calendar-day horizons because reporting is irregular. Candidate methods included:

- persistence and seven-report moving-average baselines;
- ARIMA;
- SARIMA with a seven-report seasonal period;
- simple or damped-trend exponential smoothing; and
- random forest and gradient boosting using 32 engineered features.

Validation respected time order. A final chronological holdout retained the last 20% of eligible observations, and expanding-window walk-forward evaluation trained only on outcomes known at each forecast origin. Accuracy was summarized with mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE), and weighted absolute percentage error (WAPE). MAE is emphasized because MAPE is unstable when actual discharges or net flow are zero or near zero.

## 3. Exploratory Data Analysis

### 3.1 Overall distributions

| Measure | Mean | SD | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| CBP apprehensions | 93.5 | 72.6 | 0 | 99.0 | 333 |
| Children in CBP custody | 171.5 | 126.4 | 7 | 193.0 | 531 |
| Transfers from CBP | 128.7 | 97.3 | 0 | 157.0 | 440 |
| Children in HHS care | 6,061.3 | 2,833.1 | 1,972 | 6,406.5 | 11,516 |
| HHS discharges | 173.4 | 125.7 | 0 | 181.0 | 505 |
| Net HHS flow | −44.7 | 95.9 | −465 | −11.5 | 206 |

The negative mean net flow indicates that reported discharges exceeded reported transfers by 44.7 children per observation on average. This aligns directionally with the long-run decline in HHS occupancy, but should not be treated as an accounting identity.

### 3.2 Change over time

| Year | Reports | Mean CBP apprehensions | Mean transfers | Mean HHS care | Mean discharges | Mean net flow |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 230 | 117.6 | 157.1 | 8,646.1 | 288.0 | −131.0 |
| 2024 | 251 | 148.1 | 209.4 | 7,043.0 | 205.9 | 3.4 |
| 2025 | 239 | 13.0 | 16.6 | 2,542.8 | 29.0 | −12.4 |

HHS occupancy peaked at 11,516 on 20 December 2023. The minimum, 1,972, occurred on 21 August 2025. The latest observation, 2,484 on 21 December 2025, was 78.4% below the peak but 26.0% above the August 2025 minimum. The latest series therefore shows modest recovery from its trough within a much lower-demand regime.

The annual mean fell by 18.5% from 2023 to 2024 and by 63.9% from 2024 to 2025. CBP apprehensions and transfers also changed sharply in 2025. Such structural change is more important for forecasting than small recurring seasonal effects.

Report-to-report HHS-care changes were volatile: the median change was +5 children, the standard deviation was 135, and the observed range was −888 to +720. Because reporting intervals vary, these are changes per report rather than standardized daily changes.

### 3.3 Associations among operational measures

| Pair | Pearson correlation |
|---|---:|
| CBP apprehensions and CBP custody | 0.951 |
| CBP custody and transfers | 0.925 |
| HHS care and HHS discharges | 0.921 |
| Transfers and HHS care | 0.714 |
| Apprehensions and HHS care | 0.691 |
| HHS discharges and net HHS flow | −0.644 |
| HHS care and net HHS flow | −0.483 |

The strongest relationships follow the operational pathway from apprehension to custody and transfer. The high association between HHS care and discharges is plausible because a larger care population creates more opportunities for discharge. However, shared time trends can inflate these correlations. Lagged, differenced, and policy-aware models would be required before interpreting any relationship as predictive or causal.

### 3.4 Trend and seasonality

The decomposed daily HHS-care series was dominated by trend variation: the trend component had a standard deviation of approximately 2,825 children, compared with about 45 for the seven-day seasonal component and 47 for the residual. The estimated seasonal range was roughly −70 to +55 children. This indicates that regime shifts and persistent level changes matter far more than within-week seasonality. Because decomposition required interpolation across unreported dates, the precise seasonal estimates should be regarded as exploratory.

## 4. Forecasting Results

### 4.1 Baseline behavior

For HHS care, persistence performed strongly at one report ahead in the full strict holdout (MAE 10.3; RMSE 14.1), reflecting the short-run continuity of an occupancy stock. Error increased to MAE 56.9 at seven reports and 112.2 at fourteen reports. In walk-forward testing, the corresponding persistence MAEs were 21.1, 172.0, and 313.8. The large gap between holdout and walk-forward results demonstrates sensitivity to forecast origin and regime change.

For HHS discharges and net flow, a seven-report moving average generally improved on persistence in the strict holdout. At one report, for example, discharge MAE fell from 5.64 to 4.39, while net-flow MAE fell from 7.67 to 5.80.

### 4.2 Statistical models

The following are the lowest walk-forward MAEs among the fitted statistical models in the saved evaluation results:

| Target | Horizon (reports) | Lowest-MAE statistical model | MAE | RMSE |
|---|---:|---|---:|---:|
| HHS care | 1 | SARIMA | 17.1 | 23.7 |
| HHS care | 7 | Damped Holt smoothing | 127.6 | 255.7 |
| HHS care | 14 | Damped Holt smoothing | 257.2 | 506.5 |
| HHS discharges | 1 | Simple exponential smoothing | 13.9 | 20.5 |
| HHS discharges | 7 | SARIMA | 12.6 | 18.8 |
| HHS discharges | 14 | SARIMA | 19.3 | 32.1 |
| Net HHS flow | 1 | Simple exponential smoothing | 15.2 | 25.8 |
| Net HHS flow | 7 | SARIMA | 19.2 | 28.9 |
| Net HHS flow | 14 | SARIMA | 16.4 | 22.6 |

These rankings require caution. ARIMA and exponential-smoothing walk-forward results contain 24–25 forecast origins, whereas the saved SARIMA walk-forward evaluation is sampled to 13. The results are therefore not a perfectly matched tournament. The table identifies the lowest recorded error, not a statistically proven winner.

### 4.3 Machine learning

Random forest substantially outperformed gradient boosting in many strict-holdout tests, but the machine-learning evaluation used only four strict-holdout observations and three sampled walk-forward observations per target–horizon pair. At seven and fourteen reports, machine-learning errors for HHS care were much larger than those of simpler methods. Feature importance was dominated by current HHS care and its first lag, showing that the models primarily learned persistence in the occupancy level. Given the tiny evaluation samples and strong regime shift, the machine-learning results are insufficient for production selection.

### 4.4 Interpretation

Three findings are operationally important. First, one-report HHS-care forecasts can be reasonably accurate because occupancy is persistent. Second, uncertainty expands rapidly with the horizon; a fourteen-report point forecast should not be used as a precise staffing target. Third, greater model complexity did not reliably improve generalization. In a short, shifting series, transparent statistical forecasts and strong baselines are preferable until more post-shift observations accumulate.

## 5. Key Insights

1. **The system entered a new demand regime.** The fall in mean HHS care from 8,646 in 2023 to 2,543 in 2025 is too large to treat as ordinary seasonal fluctuation.
2. **Occupancy remains path-dependent.** Current HHS care and its recent lags dominate predictive information, explaining why persistence and damped-trend models are competitive.
3. **Flows are harder to forecast proportionally than occupancy.** Discharges and net flow can be zero or close to zero, making percentage errors large and misleading even when absolute errors are operationally modest.
4. **Long-horizon forecasts are fragile.** HHS-care walk-forward MAE increases from roughly 17–27 children at one report to roughly 257–314 at fourteen reports for the better transparent models and baseline.
5. **Reporting gaps are analytically meaningful.** A “seven-report” horizon is not necessarily seven calendar days. Capacity decisions need both a reporting-index forecast and a calendar-time translation.
6. **The evidence supports association, not causation.** High correlations among apprehensions, transfers, occupancy, and discharges reflect both operational linkage and common trends; they do not estimate policy effects.

## 6. Recommendations

### 6.1 Operational recommendations

1. **Use a forecast ensemble with a baseline guardrail.** Combine persistence, damped Holt, and SARIMA forecasts, and require a candidate model to beat persistence over recent matched walk-forward origins before deployment.
2. **Plan with intervals and scenarios.** Publish a central forecast plus low- and high-demand scenarios. Tie bed, staffing, and transport decisions to the upper scenario rather than the point forecast alone.
3. **Refresh frequently.** Refit after every new report or at least weekly, and monitor recent rolling MAE because historical average performance can conceal failure after a regime shift.
4. **Separate horizons by decision.** Use one-report forecasts for immediate staffing and placement coordination; use seven- and fourteen-report forecasts only for contingency planning, procurement, and surge readiness.
5. **Monitor leading indicators.** Track CBP custody and transfers alongside HHS occupancy. Their strong associations make them useful warning signals, even though they should not be interpreted causally.
6. **Create explicit operational triggers.** Example triggers could activate review when the upper forecast interval approaches a chosen share of staffed capacity, when observed occupancy breaches the interval, or when rolling error exceeds an agreed tolerance. Actual thresholds should be set with facility-level capacity and service standards, which are absent from this dataset.

### 6.2 Analytical recommendations

1. **Standardize validation samples.** Evaluate every model on exactly the same forecast origins and report uncertainty around error differences. The current sampled SARIMA and machine-learning tests are not directly comparable with the full walk-forward results.
2. **Model calendar time explicitly.** Develop parallel forecasts in calendar days, using elapsed time as an exposure or irregular-time feature, while retaining the no-imputation reporting calendar.
3. **Detect structural breaks.** Add rolling change-point monitoring or regime indicators, and weight recent data more heavily when the process shifts.
4. **Reconcile stock and flow forecasts.** Where definitions permit, constrain predicted HHS occupancy to be coherent with prior occupancy, inflows, discharges, and documented adjustments.
5. **Improve explanatory coverage.** Add policy dates, border encounters, processing delays, sponsor-processing measures, facility capacity, length of stay, geography, and calendar effects. Without these variables, forecasts are largely autoregressive.
6. **Use MAE and interval coverage as primary metrics.** Retain RMSE to expose large misses, but avoid using MAPE as the main score for net flow or discharge series containing small denominators.
7. **Establish reproducibility controls.** Version the source extract, record retrieval dates and definitions, preserve model configurations, and produce a model card documenting intended use, error by regime, and limitations.

## 7. Limitations

The study has several limitations. The dataset is aggregated and cannot explain variation by facility, geography, age, vulnerability, length of stay, or sponsor-processing stage. Reporting is irregular, and counts may refer to different operational windows. The constructed net-flow measure may omit other movements or timing adjustments. The analysis covers fewer than three years and contains a major structural shift, limiting the stability of estimated seasonality and model parameters. Correlation analysis does not remove time trends or establish causality. Some model classes were evaluated on different and very small samples. Finally, no staffed-capacity or service-level data were available, so the study forecasts demand indicators rather than directly estimating shortages.

## 8. Conclusion

UAC care demand changed dramatically during the study period. HHS occupancy declined from a late-2023 peak above 11,500 to approximately 2,500 by December 2025, while CBP transfers and HHS discharges also fell sharply. This structural change dominates the modest weekly seasonal pattern and makes distant forecasts uncertain. Chronological validation shows that transparent statistical methods and simple baselines are more defensible than the tested machine-learning models, particularly given limited evaluation samples. The strongest operational strategy is therefore not reliance on a single sophisticated point forecast, but a frequently refreshed, short-horizon ensemble accompanied by prediction intervals, scenario thresholds, matched backtesting, and explicit monitoring for regime change.

## References

- U.S. Department of Health and Human Services. *Unaccompanied Children Information*. https://www.hhs.gov/programs/social-services/unaccompanied-children/index.html
- U.S. Department of Health and Human Services, Administration for Children and Families. *Fiscal Year 2025 Administration for Children and Families Justification of Estimates for Appropriations Committees*. https://www.acf.hhs.gov/sites/default/files/documents/olab/fy-2025-congressional-justification.pdf
- U.S. Department of Health and Human Services. *Latest UC Data—FY2024*. https://www.hhs.gov/programs/social-services/unaccompanied-children/latest-uc-data-fy2024/index.html
- Project dataset. `Data/Raw_Data.csv`, processed into 720 validated reported observations covering 12 January 2023–21 December 2025.

## Appendix A: Reproducibility Notes

The analysis-ready observations are stored in `Data/Processed/clean/uac_observed_clean.csv`; the reporting calendar is in `Data/Processed/clean/uac_reporting_calendar.csv`; decomposition output is in `Data/Processed/analysis/hhs_care_daily_decomposition.csv`; and validation and model results are under `Data/Processed/validation/` and `Data/Processed/models/`. The dashboard entry point is `app.py`.
