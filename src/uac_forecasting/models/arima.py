"""Univariate ARIMA forecasting for HHS care, discharges, and net flow."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from uac_forecasting.baseline_models import calculate_metrics
from uac_forecasting.features import FORECAST_TARGETS
from uac_forecasting.preprocess import MODELS_DIR, load_and_clean
from uac_forecasting.validation import (
    FORECAST_HORIZONS,
    prepare_horizon_data,
    strict_time_split,
    walk_forward_plan,
)


OUTPUT_DIR = MODELS_DIR / "statistical"
METRICS_PATH = OUTPUT_DIR / "arima_evaluation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "arima_evaluation_predictions.csv"
ORDERS_PATH = OUTPUT_DIR / "arima_selected_orders.csv"
ORDER_CANDIDATES = (
    (0, 0, 1),
    (1, 0, 0),
    (1, 0, 1),
    (2, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
)
HOLDOUT_EVALUATION_STEP_REPORTS = 14


def select_arima_order(series: pd.Series) -> tuple[tuple[int, int, int], float]:
    """Choose a compact ARIMA order by AIC using training history only."""
    best_order: tuple[int, int, int] | None = None
    best_aic = float("inf")
    for order in ORDER_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                fitted = ARIMA(series, order=order, trend="n").fit()
            if fitted.aic < best_aic:
                best_order, best_aic = order, float(fitted.aic)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    if best_order is None:
        raise RuntimeError("No ARIMA candidate could be fitted.")
    return best_order, best_aic


def forecast_from_origin(series: pd.Series, origin: pd.Timestamp, horizon: int, order: tuple[int, int, int]) -> float:
    """Fit using values available at the origin and forecast future reports."""
    history = series.loc[:origin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(history, order=order, trend="n").fit()
        return float(fitted.forecast(steps=horizon).iloc[-1])


def _evaluate_origins(
    data: pd.DataFrame,
    series: pd.Series,
    target_name: str,
    horizon: int,
    order: tuple[int, int, int],
    origins: pd.DatetimeIndex,
    strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    target_column = f"target_{target_name}_next_{horizon}_report"
    subset = data.loc[origins]
    predictions = [forecast_from_origin(series, date, horizon, order) for date in origins]
    results = pd.DataFrame(
        {
            "strategy": strategy,
            "target": target_name,
            "horizon_reports": horizon,
            "model": f"arima_{order}",
            "forecast_origin_date": origins,
            "target_date": subset["target_date"].to_numpy(),
            "actual_value": subset[target_column].to_numpy(),
            "predicted_value": predictions,
        }
    )
    metrics = {
        "strategy": strategy,
        "target": target_name,
        "horizon_reports": horizon,
        "model": f"arima_{order}",
        "arima_order": str(order),
        **calculate_metrics(results["actual_value"], results["predicted_value"]),
    }
    return metrics, results


def sampled_holdout_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Limit costly ARIMA refits while retaining coverage of the final period."""
    sampled = index[::HOLDOUT_EVALUATION_STEP_REPORTS]
    if sampled[-1] != index[-1]:
        sampled = sampled.append(pd.DatetimeIndex([index[-1]]))
    return sampled


def evaluate_arima() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate selected univariate ARIMA models across all targets/horizons."""
    observed = load_and_clean()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_orders: list[dict[str, object]] = []

    for target_name in FORECAST_TARGETS:
        series = observed[target_name]
        # The earliest walk-forward origin is the only data used for order selection.
        first_data = prepare_horizon_data(target_name, FORECAST_HORIZONS[0])
        first_folds = walk_forward_plan(first_data, target_name, FORECAST_HORIZONS[0])
        selection_end = pd.Timestamp(first_folds.iloc[0]["forecast_origin_date"])
        order, aic = select_arima_order(series.loc[:selection_end])
        selected_orders.append(
            {
                "target": target_name,
                "selection_end_date": selection_end,
                "arima_order": str(order),
                "aic": aic,
            }
        )

        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            _, holdout = strict_time_split(data)
            holdout_origins = sampled_holdout_origins(holdout.index)
            metrics, predictions = _evaluate_origins(
                data, series, target_name, horizon, order, holdout_origins, "strict_holdout_sampled"
            )
            metric_rows.append(metrics)
            prediction_rows.append(predictions)

            folds = walk_forward_plan(data, target_name, horizon)
            origins = pd.DatetimeIndex(folds["forecast_origin_date"])
            metrics, predictions = _evaluate_origins(
                data, series, target_name, horizon, order, origins, "walk_forward"
            )
            metric_rows.append(metrics)
            prediction_rows.append(predictions)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(selected_orders),
    )


def run_arima_evaluation() -> tuple[Path, Path, Path]:
    """Write ARIMA metrics, predictions, and AIC-selected orders."""
    metrics, predictions, orders = evaluate_arima()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    orders.to_csv(ORDERS_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(metrics)} metric rows to {METRICS_PATH}")
    print("Selected orders:")
    print(orders.round(2).to_string(index=False))
    return METRICS_PATH, PREDICTIONS_PATH, ORDERS_PATH


if __name__ == "__main__":
    run_arima_evaluation()
