"""Backward-compatible validation command."""

from uac_forecasting.evaluation.validation import *  # noqa: F403
from uac_forecasting.evaluation.validation import run_validation_setup

if __name__ == "__main__":
    run_validation_setup()
