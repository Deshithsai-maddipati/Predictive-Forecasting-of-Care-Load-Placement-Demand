"""Univariate Exponential Smoothing forecasts for UAC operational targets."""

from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

from uac_forecasting.arima_models import sampled_holdout_origins
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
METRICS_PATH = OUTPUT_DIR / "exponential_smoothing_evaluation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "exponential_smoothing_evaluation_predictions.csv"
MODELS_PATH = OUTPUT_DIR / "exponential_smoothing_selected_models.csv"
MODEL_CANDIDATES = ("simple", "holt", "holt_damped")


def fit_exponential_smoothing(series: pd.Series, model_type: str):
    """Fit one non-seasonal smoothing variant to a reported-observation series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_type == "simple":
            return SimpleExpSmoothing(series, initialization_method="estimated").fit(
                optimized=True
            )
        if model_type == "holt":
            return Holt(series, initialization_method="estimated", damped_trend=False).fit(
                optimized=True
            )
        if model_type == "holt_damped":
            return Holt(series, initialization_method="estimated", damped_trend=True).fit(
                optimized=True
            )
    raise ValueError(f"Unknown model_type: {model_type}")


def select_model(series: pd.Series) -> tuple[str, float]:
    """Select the best smoothing structure by AIC on historical training data."""
    results: list[tuple[str, float]] = []
    for model_type in MODEL_CANDIDATES:
        try:
            fitted = fit_exponential_smoothing(series, model_type)
            results.append((model_type, float(fitted.aic)))
        except (ValueError, RuntimeError):
            continue
    if not results:
        raise RuntimeError("No Exponential Smoothing candidate could be fitted.")
    return min(results, key=lambda result: result[1])


def forecast_from_origin(
    series: pd.Series, origin: pd.Timestamp, horizon: int, model_type: str
) -> float:
    """Fit only observed history through the origin and forecast ahead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit_exponential_smoothing(series.loc[:origin], model_type)
        return float(fitted.forecast(horizon).iloc[-1])


def _evaluate_origins(
    data: pd.DataFrame,
    series: pd.Series,
    target_name: str,
    horizon: int,
    model_type: str,
    origins: pd.DatetimeIndex,
    strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    target_column = f"target_{target_name}_next_{horizon}_report"
    subset = data.loc[origins]
    predicted = [forecast_from_origin(series, origin, horizon, model_type) for origin in origins]
    results = pd.DataFrame(
        {
            "strategy": strategy,
            "target": target_name,
            "horizon_reports": horizon,
            "model": f"exponential_smoothing_{model_type}",
            "forecast_origin_date": origins,
            "target_date": subset["target_date"].to_numpy(),
            "actual_value": subset[target_column].to_numpy(),
            "predicted_value": predicted,
        }
    )
    metrics = {
        "strategy": strategy,
        "target": target_name,
        "horizon_reports": horizon,
        "model": f"exponential_smoothing_{model_type}",
        "smoothing_model": model_type,
        **calculate_metrics(results["actual_value"], results["predicted_value"]),
    }
    return metrics, results


def evaluate_exponential_smoothing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate selected smoothing models across target and horizon combinations."""
    observed = load_and_clean()
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    selected_models: list[dict[str, object]] = []

    for target_name in FORECAST_TARGETS:
        series = observed[target_name]
        first_data = prepare_horizon_data(target_name, FORECAST_HORIZONS[0])
        first_folds = walk_forward_plan(first_data, target_name, FORECAST_HORIZONS[0])
        selection_end = pd.Timestamp(first_folds.iloc[0]["forecast_origin_date"])
        model_type, aic = select_model(series.loc[:selection_end])
        selected_models.append(
            {
                "target": target_name,
                "selection_end_date": selection_end,
                "smoothing_model": model_type,
                "aic": aic,
            }
        )

        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            _, holdout = strict_time_split(data)
            metric, result = _evaluate_origins(
                data,
                series,
                target_name,
                horizon,
                model_type,
                sampled_holdout_origins(holdout.index),
                "strict_holdout_sampled",
            )
            metrics.append(metric)
            predictions.append(result)

            folds = walk_forward_plan(data, target_name, horizon)
            metric, result = _evaluate_origins(
                data,
                series,
                target_name,
                horizon,
                model_type,
                pd.DatetimeIndex(folds["forecast_origin_date"]),
                "walk_forward",
            )
            metrics.append(metric)
            predictions.append(result)

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(selected_models)


def run_exponential_smoothing_evaluation() -> tuple[Path, Path, Path]:
    """Write selected models, forecasts, and forecast errors."""
    metrics, predictions, models = evaluate_exponential_smoothing()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    models.to_csv(MODELS_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(metrics)} metric rows to {METRICS_PATH}")
    print("Selected smoothing models:")
    print(models.round(2).to_string(index=False))
    return METRICS_PATH, PREDICTIONS_PATH, MODELS_PATH


if __name__ == "__main__":
    run_exponential_smoothing_evaluation()
