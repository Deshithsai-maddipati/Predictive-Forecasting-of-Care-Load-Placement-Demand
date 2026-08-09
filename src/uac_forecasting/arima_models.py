"""Backward-compatible ARIMA command."""

from uac_forecasting.models.arima import *  # noqa: F403
from uac_forecasting.models.arima import run_arima_evaluation

if __name__ == "__main__":
    run_arima_evaluation()
