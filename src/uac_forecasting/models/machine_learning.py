"""Tree-based forecasting models using leakage-safe engineered features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from uac_forecasting.arima_models import sampled_holdout_origins
from uac_forecasting.baseline_models import calculate_metrics
from uac_forecasting.features import FORECAST_TARGETS
from uac_forecasting.preprocess import MODELS_DIR
from uac_forecasting.validation import (
    FORECAST_HORIZONS,
    prepare_horizon_data,
    strict_time_split,
    walk_forward_plan,
)


OUTPUT_DIR = MODELS_DIR / "machine_learning"
METRICS_PATH = OUTPUT_DIR / "ml_evaluation_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "ml_evaluation_predictions.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "ml_feature_importance.csv"
WALK_FORWARD_EVALUATION_STEP_FOLDS = 14


def predictor_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric predictors, excluding every future target and metadata."""
    excluded = {"target_date"}
    return [
        column
        for column in data.select_dtypes(include="number").columns
        if column not in excluded and not column.startswith("target_")
    ]


def build_model(model_name: str):
    """Create deterministic tree-model configurations for the comparison."""
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=20,
            max_features=0.8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=30,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=42,
            loss="squared_error",
        )
    raise ValueError(f"Unknown model: {model_name}")


def sampled_walk_forward_origins(folds: pd.DataFrame) -> pd.DatetimeIndex:
    """Reduce expensive repeated refits while preserving chronological spacing."""
    origins = pd.DatetimeIndex(folds["forecast_origin_date"])
    sampled = origins[::WALK_FORWARD_EVALUATION_STEP_FOLDS]
    if sampled[-1] != origins[-1]:
        sampled = sampled.append(pd.DatetimeIndex([origins[-1]]))
    return sampled


def sampled_holdout_ml_origins(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Use every second ARIMA holdout origin for feasible tree-model refits."""
    origins = sampled_holdout_origins(index)
    sampled = origins[::4]
    if sampled[-1] != origins[-1]:
        sampled = sampled.append(pd.DatetimeIndex([origins[-1]]))
    return sampled


def _evaluate_origins(
    data: pd.DataFrame,
    target_name: str,
    horizon: int,
    origins: pd.DatetimeIndex,
    strategy: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    target_column = f"target_{target_name}_next_{horizon}_report"
    features = predictor_columns(data)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name in ("random_forest", "gradient_boosting"):
        rows: list[dict[str, object]] = []
        for origin in origins:
            train = data.loc[data["target_date"] < origin]
            model = build_model(model_name)
            model.fit(train[features], train[target_column])
            test_row = data.loc[[origin]]
            rows.append(
                {
                    "strategy": strategy,
                    "target": target_name,
                    "horizon_reports": horizon,
                    "model": model_name,
                    "forecast_origin_date": origin,
                    "target_date": test_row.iloc[0]["target_date"],
                    "actual_value": test_row.iloc[0][target_column],
                    "predicted_value": float(model.predict(test_row[features])[0]),
                    "training_rows": len(train),
                }
            )
        predictions = pd.DataFrame(rows)
        metric_rows.append(
            {
                "strategy": strategy,
                "target": target_name,
                "horizon_reports": horizon,
                "model": model_name,
                "feature_count": len(features),
                **calculate_metrics(predictions["actual_value"], predictions["predicted_value"]),
            }
        )
        prediction_rows.append(predictions)
    return metric_rows, prediction_rows


def feature_importance(data: pd.DataFrame, target_name: str, horizon: int) -> pd.DataFrame:
    """Fit final models on all labelled rows and return feature importances."""
    target_column = f"target_{target_name}_next_{horizon}_report"
    features = predictor_columns(data)
    rows: list[pd.DataFrame] = []
    for model_name in ("random_forest", "gradient_boosting"):
        model = build_model(model_name)
        model.fit(data[features], data[target_column])
        rows.append(
            pd.DataFrame(
                {
                    "target": target_name,
                    "horizon_reports": horizon,
                    "model": model_name,
                    "feature": features,
                    "importance": model.feature_importances_,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def evaluate_ml_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate feature-based ML models for every target and horizon."""
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    for target_name in FORECAST_TARGETS:
        for horizon in FORECAST_HORIZONS:
            data = prepare_horizon_data(target_name, horizon)
            _, holdout = strict_time_split(data)
            metric_rows, prediction_rows = _evaluate_origins(
                data,
                target_name,
                horizon,
                sampled_holdout_ml_origins(holdout.index),
                "strict_holdout_sampled",
            )
            metrics.extend(metric_rows)
            predictions.extend(prediction_rows)

            folds = walk_forward_plan(data, target_name, horizon)
            metric_rows, prediction_rows = _evaluate_origins(
                data,
                target_name,
                horizon,
                sampled_walk_forward_origins(folds),
                "walk_forward_sampled",
            )
            metrics.extend(metric_rows)
            predictions.extend(prediction_rows)
            importances.append(feature_importance(data, target_name, horizon))

    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True),
        pd.concat(importances, ignore_index=True),
    )


def run_ml_evaluation() -> tuple[Path, Path, Path]:
    """Write ML model metrics, forecast predictions, and feature importance."""
    metrics, predictions, importances = evaluate_ml_models()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d")
    importances.to_csv(IMPORTANCE_PATH, index=False)
    print(f"Wrote {len(metrics)} metric rows to {METRICS_PATH}")
    return METRICS_PATH, PREDICTIONS_PATH, IMPORTANCE_PATH


if __name__ == "__main__":
    run_ml_evaluation()
