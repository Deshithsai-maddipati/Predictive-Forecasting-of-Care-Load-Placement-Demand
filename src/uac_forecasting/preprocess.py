"""Backward-compatible preprocessing command."""

from uac_forecasting.data.preprocess import *  # noqa: F403
from uac_forecasting.data.preprocess import run_preprocessing

if __name__ == "__main__":
    run_preprocessing()
