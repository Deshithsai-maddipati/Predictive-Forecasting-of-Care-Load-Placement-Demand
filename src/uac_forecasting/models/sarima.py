"""SARIMA forecasting with a seven-reported-observation seasonal cycle.

The source is not a complete daily series, so the seasonal period represents
seven reported observations rather than a confirmed seven-calendar-day cycle.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from uac_forecasting.arima_models import (
    sampled_holdout_origins,
    select_arima_order,
)
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
METRICS_PATH = OUTPUT_DIR / "sarima_evaluation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "sarima_evaluation_predictions.csv"
ORDERS_PATH = OUTPUT_DIR / "sarima_selected_orders.csv"
SEASONAL_PERIOD_REPORTS = 7
SEASONAL_CANDIDATES = (
    (1, 0, 0, SEASONAL_PERIOD_REPORTS),
    (0, 0, 1, SEASONAL_PERIOD_REPORTS),
    (1, 0, 1, SEASONAL_PERIOD_REPORTS),
)
WALK_FORWARD_EVALUATION_STEP_FOLDS = 2


def select_seasonal_order(
    series: pd.Series, order: tuple[int, int, int]
) -> tuple[tuple[int, int, int, int], float]:
    """Choose a seasonal component by AIC on pre-validation history only."""
    best_order: tuple[int, int, int, int] | None = None
    best_aic = float("inf")
    for seasonal_order in SEASONAL_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=25)
            if fitted.aic < best_aic:
                best_order, best_aic = seasonal_order, float(fitted.aic)
        except (ValueError, RuntimeError):
            continue
    if best_order is None:
        raise RuntimeError("No SARIMA seasonal candidate could be fitted.")
    return best_order, best_aic


def forecast_sarima(
    series: pd.Series,
    origin: pd.Timestamp,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> float:
    """Fit only to history available at the origin and forecast ahead."""
    history = series.loc[:origin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=25)
        return float(fitted.forecast(steps=horizon).iloc[-1])


def _evaluate_origins(
    data: pd.DataFrame,
    series: pd.Series,
    target_name: str,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    origins: pd.DatetimeIndex,
    strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    target_column = f"target_{target_name}_next_{horizon}_report"
    subset = data.loc[origins]
    predictions = [
        forecast_sarima(series, date, horizon, order, seasonal_order) for date in origins
    ]
    model_name = f"sarima_{order}x{seasonal_order}"
    results = pd.DataFrame(
        {
            "strategy": strategy,
            "target": target_name,
            "horizon_reports": horizon,
            "model": model_name,
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
        "model": model_name,
        "arima_order": str(order),
        "seasonal_order": str(seasonal_order),
        **calculate_metrics(results["actual_value"], results["predicted_value"]),
    }
    return metrics, results


def sampled_walk_forward_origins(folds: pd.DataFrame) -> pd.DatetimeIndex:
    """Sample expanding-window folds to keep repeated SARIMA refits feasible."""
    origins = pd.DatetimeIndex(folds["forecast_origin_date"])
    sampled = origins[::WALK_FORWARD_EVALUATION_STEP_FOLDS]
    if sampled[-1] != origins[-1]:
        sampled = sampled.append(pd.DatetimeIndex([origins[-1]]))
    return sampled


def evaluate_sarima() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate SARIMA across all targets and selected report horizons."""
    observed = load_and_clean()
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_orders: list[dict[str, object]] = []

    for target_name in FORECAST_TARGETS:
        series = observed[target_name]
        first_data = prepare_horizon_data(target_name, FORECAST_HORIZONS[0])
        first_folds = walk_forward_plan(first_data, target_name, FORECAST_HORIZONS[0])
        selection_end = pd.Timestamp(first_folds.iloc[0]["forecast_origin_date"])
        order, _ = select_arima_order(series.loc[:selection_end])
        seasonal_order, aic = select_seasonal_order(series.loc[:selection_end], order)
        selected_orders.append(
            {
                "target": target_name,
                "selection_end_date": selection_end,
                "arima_order": str(order),
                "seasonal_order": str(seasonal_order),
                "seasonal_period_unit": "reported observations",
                "aic": aic,
            }
        )

        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            _, holdout = strict_time_split(data)
            metrics, predictions = _evaluate_origins(
                data,
                series,
                target_name,
                horizon,
                order,
                seasonal_order,
                sampled_holdout_origins(holdout.index),
                "strict_holdout_sampled",
            )
            metric_rows.append(metrics)
            prediction_rows.append(predictions)

            folds = walk_forward_plan(data, target_name, horizon)
            metrics, predictions = _evaluate_origins(
                data,
                series,
                target_name,
                horizon,
                order,
                seasonal_order,
                sampled_walk_forward_origins(folds),
                "walk_forward_sampled",
            )
            metric_rows.append(metrics)
            prediction_rows.append(predictions)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(selected_orders),
    )


def run_sarima_evaluation() -> tuple[Path, Path, Path]:
    """Write SARIMA metrics, predictions, and selected seasonal orders."""
    metrics, predictions, orders = evaluate_sarima()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    orders.to_csv(ORDERS_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(metrics)} metric rows to {METRICS_PATH}")
    print("Selected orders:")
    print(orders.round(2).to_string(index=False))
    return METRICS_PATH, PREDICTIONS_PATH, ORDERS_PATH


if __name__ == "__main__":
    run_sarima_evaluation()
