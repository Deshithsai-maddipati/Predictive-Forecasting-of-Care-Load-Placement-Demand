"""Backward-compatible decomposition command."""

from uac_forecasting.analysis.decomposition import *  # noqa: F403
from uac_forecasting.analysis.decomposition import run_decomposition

if __name__ == "__main__":
    run_decomposition()
