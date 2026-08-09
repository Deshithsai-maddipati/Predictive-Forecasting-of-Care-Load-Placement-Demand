"""Baseline forecasts and time-respecting evaluation for UAC outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uac_forecasting.features import FORECAST_TARGETS
from uac_forecasting.validation import FORECAST_HORIZONS, prepare_horizon_data, strict_time_split, walk_forward_plan
from uac_forecasting.preprocess import MODELS_DIR


OUTPUT_DIR = MODELS_DIR / "baselines"
METRICS_PATH = OUTPUT_DIR / "baseline_evaluation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "baseline_evaluation_predictions.csv"


def baseline_predictions(data: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Return forecasting-safe predictions available at each report date."""
    return pd.DataFrame(
        {
            "naive_persistence": data[target_name],
            "moving_average_7_reports": data[f"{target_name}_rolling_mean_7_reports"],
        },
        index=data.index,
    )


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate standard forecast errors for aligned, non-null values."""
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    errors = aligned["actual"] - aligned["predicted"]
    nonzero_actual = aligned["actual"] != 0
    return {
        "observations": len(aligned),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mape_percent": float(
            (errors[nonzero_actual].abs() / aligned.loc[nonzero_actual, "actual"].abs()).mean() * 100
        ) if nonzero_actual.any() else float("nan"),
        "wape_percent": float(errors.abs().sum() / aligned["actual"].abs().sum() * 100),
    }


def _evaluate_rows(
    data: pd.DataFrame, target_name: str, origin_dates: pd.DatetimeIndex, horizon: int, strategy: str
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    target = f"target_{target_name}_next_{horizon}_report"
    subset = data.loc[origin_dates]
    forecasts = baseline_predictions(subset, target_name)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name in forecasts.columns:
        prediction = forecasts[model_name]
        metric_rows.append(
            {
                "strategy": strategy,
                "target": target_name,
                "horizon_reports": horizon,
                "model": model_name,
                **calculate_metrics(subset[target], prediction),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "strategy": strategy,
                    "target": target_name,
                    "horizon_reports": horizon,
                    "model": model_name,
                    "forecast_origin_date": subset.index,
                    "target_date": subset["target_date"].to_numpy(),
                    "actual_value": subset[target].to_numpy(),
                    "predicted_value": prediction.to_numpy(),
                }
            )
        )
    return metric_rows, prediction_rows


def evaluate_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baselines on final holdouts and expanding walk-forward folds."""
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for target_name in FORECAST_TARGETS:
        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            _, holdout_test = strict_time_split(data)
            metrics, predictions = _evaluate_rows(
                data, target_name, holdout_test.index, horizon, "strict_holdout"
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)

            folds = walk_forward_plan(data, target_name, horizon)
            origin_dates = pd.DatetimeIndex(folds["forecast_origin_date"])
            metrics, predictions = _evaluate_rows(
                data, target_name, origin_dates, horizon, "walk_forward"
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)

    return pd.DataFrame(all_metrics), pd.concat(all_predictions, ignore_index=True)


def run_baseline_evaluation() -> tuple[Path, Path]:
    """Write baseline forecast metrics and per-forecast predictions."""
    metrics, predictions = evaluate_baselines()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(metrics)} metric rows to {METRICS_PATH}")
    print(metrics[["strategy", "target", "horizon_reports", "model", "mae", "rmse", "wape_percent"]].round(2).to_string(index=False))
    return METRICS_PATH, PREDICTIONS_PATH


if __name__ == "__main__":
    run_baseline_evaluation()
