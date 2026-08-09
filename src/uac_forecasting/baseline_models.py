"""Backward-compatible baseline-model command."""

from uac_forecasting.models.baselines import *  # noqa: F403
from uac_forecasting.models.baselines import run_baseline_evaluation

if __name__ == "__main__":
    run_baseline_evaluation()
