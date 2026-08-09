"""Backward-compatible SARIMA command."""

from uac_forecasting.models.sarima import *  # noqa: F403
from uac_forecasting.models.sarima import run_sarima_evaluation

if __name__ == "__main__":
    run_sarima_evaluation()
