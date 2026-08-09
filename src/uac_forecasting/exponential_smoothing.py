"""Backward-compatible Exponential Smoothing command."""

from uac_forecasting.models.exponential_smoothing import *  # noqa: F403
from uac_forecasting.models.exponential_smoothing import run_exponential_smoothing_evaluation

if __name__ == "__main__":
    run_exponential_smoothing_evaluation()
