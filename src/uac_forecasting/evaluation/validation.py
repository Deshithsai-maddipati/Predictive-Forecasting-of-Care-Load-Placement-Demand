"""Time-respecting train/test and walk-forward validation plans."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uac_forecasting.features import FORECAST_TARGETS, build_features
from uac_forecasting.preprocess import VALIDATION_DIR


VALIDATION_PLAN_PATH = VALIDATION_DIR / "time_validation_plan.csv"
FORECAST_HORIZONS = (1, 7, 14)  # future reported observations


def prepare_horizon_data(target_name: str, horizon: int) -> pd.DataFrame:
    """Build complete, chronological samples for one target and horizon."""
    if target_name not in FORECAST_TARGETS:
        raise ValueError(f"target_name must be one of {FORECAST_TARGETS}.")
    frame = build_features(forecast_horizon=horizon).copy()
    target_column = f"target_{target_name}_next_{horizon}_report"
    frame["target_date"] = pd.Series(frame.index, index=frame.index).shift(-horizon)

    required = [f"{target_name}_lag_14_reports", target_column, "target_date"]
    return frame.dropna(subset=required).sort_index()


def strict_time_split(data: pd.DataFrame, test_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically, keeping all test forecast origins after train data."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    test_start_position = int(len(data) * (1 - test_fraction))
    test_start_date = data.index[test_start_position]
    # Only train targets already observable before the final holdout begins.
    train = data.loc[data["target_date"] < test_start_date].copy()
    test = data.iloc[test_start_position:].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train, test


def walk_forward_plan(
    data: pd.DataFrame,
    target_name: str,
    horizon: int,
    initial_train_reports: int = 365,
    step_reports: int = 14,
) -> pd.DataFrame:
    """Return expanding-window forecast origins for a single horizon.

    Each fold trains only on samples whose target was known at the forecast
    origin.  The next fold expands the train window; it never shuffles data.
    """
    if initial_train_reports < 1 or step_reports < 1:
        raise ValueError("initial_train_reports and step_reports must be positive.")

    rows: list[dict[str, object]] = []
    for position in range(initial_train_reports, len(data), step_reports):
        origin_date = data.index[position]
        train = data.loc[data["target_date"] < origin_date]
        target_date = data.iloc[position]["target_date"]
        if pd.isna(target_date) or train.empty:
            continue
        rows.append(
            {
                "strategy": "walk_forward",
                "target": target_name,
                "horizon_reports": horizon,
                "fold": len(rows) + 1,
                "train_start_date": train.index.min(),
                "train_end_date": train.index.max(),
                "forecast_origin_date": origin_date,
                "target_date": target_date,
                "training_rows": len(train),
                "test_rows": 1,
            }
        )
    return pd.DataFrame(rows)


def create_validation_plan() -> pd.DataFrame:
    """Create chronological holdout and walk-forward plans for all horizons."""
    plans: list[pd.DataFrame] = []
    for target_name in FORECAST_TARGETS:
        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            train, test = strict_time_split(data)
            holdout = pd.DataFrame(
                [
                    {
                        "strategy": "strict_holdout",
                        "target": target_name,
                        "horizon_reports": horizon,
                        "fold": "final",
                        "train_start_date": train.index.min(),
                        "train_end_date": train.index.max(),
                        "forecast_origin_date": test.index.min(),
                        "target_date": test.iloc[0]["target_date"],
                        "training_rows": len(train),
                        "test_rows": len(test),
                    }
                ]
            )
            plans.extend([holdout, walk_forward_plan(data, target_name, horizon)])
    return pd.concat(plans, ignore_index=True)


def run_validation_setup() -> Path:
    """Write the validation plan used by all forecasting models."""
    plan = create_validation_plan()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(VALIDATION_PLAN_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(plan)} validation rows to {VALIDATION_PLAN_PATH}")
    print(plan.groupby(["strategy", "target", "horizon_reports"]).size().to_string())
    return VALIDATION_PLAN_PATH


if __name__ == "__main__":
    run_validation_setup()
