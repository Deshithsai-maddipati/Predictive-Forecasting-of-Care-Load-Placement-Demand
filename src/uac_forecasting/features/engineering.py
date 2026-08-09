"""Create leakage-safe predictive features from reported UAC observations.

The source has systematic non-reporting dates.  Therefore, lag and rolling
windows are measured in *reported observations*, not assumed calendar days.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from uac_forecasting.preprocess import FEATURES_DIR, load_and_clean


FEATURE_PATH = FEATURES_DIR / "uac_forecasting_features.csv"
LAG_PERIODS = (1, 7, 14)
ROLLING_WINDOWS = (7, 14)
FORECAST_TARGETS = ("hhs_care", "hhs_discharges", "net_hhs_flow")


def _holiday_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return USFederalHolidayCalendar().holidays(start=start, end=end)


def build_features(forecast_horizon: int = 1) -> pd.DataFrame:
    """Return features and targets for the next reported forecasting period.

    A feature row at time *t* uses no future information.  The target is HHS
    care at the next ``forecast_horizon`` reported observation(s).
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")

    frame = load_and_clean().copy()
    for measure in FORECAST_TARGETS:
        series = frame[measure]
        for lag in LAG_PERIODS:
            frame[f"{measure}_lag_{lag}_reports"] = series.shift(lag)

        # Shift first: no row can use its own target value to predict the future.
        for window in ROLLING_WINDOWS:
            prior_values = series.shift(1).rolling(window=window, min_periods=window)
            frame[f"{measure}_rolling_mean_{window}_reports"] = prior_values.mean()
            frame[f"{measure}_rolling_variance_{window}_reports"] = prior_values.var()

    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    holidays = _holiday_dates(frame.index.min(), frame.index.max())
    frame["is_us_federal_holiday"] = frame.index.normalize().isin(holidays).astype(int)

    for measure in FORECAST_TARGETS:
        frame[f"target_{measure}_next_{forecast_horizon}_report"] = frame[measure].shift(
            -forecast_horizon
        )
    return frame


def run_feature_engineering() -> Path:
    """Write a ready-to-model feature table for one-report-ahead forecasts."""
    features = build_features(forecast_horizon=1)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_PATH, index=True, date_format="%Y-%m-%d")

    model_ready = features.dropna(
        subset=["hhs_care_lag_14_reports", "target_hhs_care_next_1_report"]
    )
    print(f"Wrote {len(features)} feature rows to {FEATURE_PATH}")
    print(f"Model-ready rows after lag/target availability: {len(model_ready)}")
    return FEATURE_PATH


if __name__ == "__main__":
    run_feature_engineering()
