"""Backward-compatible machine-learning command."""

from uac_forecasting.models.machine_learning import *  # noqa: F403
from uac_forecasting.models.machine_learning import run_ml_evaluation

if __name__ == "__main__":
    run_ml_evaluation()
