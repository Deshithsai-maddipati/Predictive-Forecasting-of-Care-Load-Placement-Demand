"""Decompose the HHS care series into trend, seasonality, and residuals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose

from uac_forecasting.preprocess import ANALYSIS_DIR, load_and_clean


DECOMPOSITION_PATH = ANALYSIS_DIR / "hhs_care_daily_decomposition.csv"
WEEKLY_PERIOD_DAYS = 7


def decompose_hhs_care() -> pd.DataFrame:
    """Build a daily, additive decomposition of the HHS-care census.

    Interpolation is used only to support this exploratory decomposition.
    ``is_observed`` retains the distinction between reported and inferred dates.
    """
    observed = load_and_clean()["hhs_care"]
    daily = observed.asfreq("D")
    interpolated = daily.interpolate(method="time", limit_area="inside")

    result: DecomposeResult = seasonal_decompose(
        interpolated,
        model="additive",
        period=WEEKLY_PERIOD_DAYS,
        extrapolate_trend="freq",
    )
    return pd.DataFrame(
        {
            "is_observed": daily.notna(),
            "hhs_care_observed": daily,
            "hhs_care_interpolated": interpolated,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    )


def run_decomposition() -> Path:
    """Write decomposition components for exploratory analysis."""
    components = decompose_hhs_care()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    components.to_csv(DECOMPOSITION_PATH, index=True, date_format="%Y-%m-%d")

    print(f"Wrote {len(components)} daily decomposition rows to {DECOMPOSITION_PATH}")
    print(f"Interpolated dates used only for decomposition: {(~components['is_observed']).sum()}")
    print(f"Trend range: {components['trend'].min():.0f} to {components['trend'].max():.0f}")
    print(f"Residual standard deviation: {components['residual'].std():.1f}")
    return DECOMPOSITION_PATH


if __name__ == "__main__":
    run_decomposition()
