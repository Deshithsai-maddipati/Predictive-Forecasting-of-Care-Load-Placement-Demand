# UAC Care Forecasting

Short-term forecasting and capacity-risk analysis for children in HHS care.

## Project layout

```text
Data/Processed/
  clean/             validated observations and reporting calendar
  analysis/          decomposition output
  features/          forecasting feature table
  validation/        time-based validation plans
  models/            baseline, statistical, and ML results
src/uac_forecasting/
  data/              preprocessing pipeline
  analysis/          decomposition utilities
  features/          feature engineering
  evaluation/        validation strategy
  models/            baseline, ARIMA, SARIMA, smoothing, and ML models
app.py               Streamlit dashboard
```

## Setup and dashboard

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
streamlit run app.py
```
