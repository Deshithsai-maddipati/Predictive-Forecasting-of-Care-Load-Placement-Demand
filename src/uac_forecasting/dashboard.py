"""Shared presentation and forecasting helpers for the Streamlit pages."""

from __future__ import annotations

import ast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from uac_forecasting.data.preprocess import MODELS_DIR
from uac_forecasting.models.arima import forecast_from_origin as arima_forecast
from uac_forecasting.models.exponential_smoothing import forecast_from_origin as smoothing_forecast
from uac_forecasting.models.sarima import forecast_sarima
from uac_forecasting.models.machine_learning import build_model, predictor_columns
from uac_forecasting.features import build_features


MODEL_OPTIONS = (
    "Exponential Smoothing",
    "Naive Persistence",
    "ARIMA",
    "SARIMA",
    "Random Forest",
    "Gradient Boosting",
)
TARGET_LABELS = {
    "hhs_care": "Children in HHS Care",
    "hhs_discharges": "HHS Discharges",
    "net_hhs_flow": "Net HHS Flow",
}


def apply_dashboard_style() -> None:
    """Apply the shared visual styling across every Streamlit page."""
    st.markdown(
        """<style>
        .stApp { background: linear-gradient(150deg, #f8fafc, #eef6ff 55%, #f8fafc); }
        [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px; box-shadow: 0 2px 8px rgba(15,23,42,.05); }
        h1, h2, h3 { color: #0f172a; }
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data
def load_metrics() -> pd.DataFrame:
    paths = {
        "Naive Persistence": MODELS_DIR / "baselines" / "baseline_evaluation_metrics.csv",
        "ARIMA": MODELS_DIR / "statistical" / "arima_evaluation_metrics.csv",
        "SARIMA": MODELS_DIR / "statistical" / "sarima_evaluation_metrics.csv",
        "Exponential Smoothing": MODELS_DIR / "statistical" / "exponential_smoothing_evaluation_metrics.csv",
    }
    frames = []
    for display_model, path in paths.items():
        if path.exists():
            frame = pd.read_csv(path)
            frame["display_model"] = display_model
            frame["model_label"] = display_model
            if display_model == "Naive Persistence":
                frame["model_label"] = frame["model"].map(
                    {
                        "naive_persistence": "Naive persistence",
                        "moving_average_7_reports": "Moving average (7 reports)",
                    }
                )
            frames.append(frame)
    ml_path = MODELS_DIR / "machine_learning" / "ml_evaluation_metrics.csv"
    if ml_path.exists():
        ml_metrics = pd.read_csv(ml_path)
        for display_model, internal_name in {
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting",
        }.items():
            frame = ml_metrics.loc[ml_metrics["model"].eq(internal_name)].copy()
            frame["display_model"] = display_model
            frame["model_label"] = display_model
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_selected_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = MODELS_DIR / "statistical"
    return (
        pd.read_csv(output / "exponential_smoothing_selected_models.csv"),
        pd.read_csv(output / "arima_selected_orders.csv"),
        pd.read_csv(output / "sarima_selected_orders.csv"),
    )


def _selected_row(frame: pd.DataFrame, target: str) -> pd.Series:
    return frame.loc[frame["target"].eq(target)].iloc[0]


def make_forecast(target: str, horizon: int, model_name: str, observed: pd.DataFrame) -> float:
    """Forecast one target with the selected model family."""
    series = observed[target]
    if model_name == "Naive Persistence":
        return float(series.iloc[-1])
    smoothing, arima, sarima = load_selected_models()
    if model_name == "Exponential Smoothing":
        return smoothing_forecast(series, series.index[-1], horizon, _selected_row(smoothing, target)["smoothing_model"])
    if model_name == "ARIMA":
        return arima_forecast(series, series.index[-1], horizon, ast.literal_eval(_selected_row(arima, target)["arima_order"]))
    if model_name in {"Random Forest", "Gradient Boosting"}:
        internal_name = "random_forest" if model_name == "Random Forest" else "gradient_boosting"
        features = build_features(forecast_horizon=horizon)
        target_column = f"target_{target}_next_{horizon}_report"
        columns = predictor_columns(features)
        training = features.dropna(subset=[target_column, *columns]).copy()
        model = build_model(internal_name)
        model.fit(training[columns], training[target_column])
        latest_features = features.loc[[features.index[-1]], columns]
        return float(model.predict(latest_features)[0])
    row = _selected_row(sarima, target)
    return forecast_sarima(series, series.index[-1], horizon, ast.literal_eval(row["arima_order"]), ast.literal_eval(row["seasonal_order"]))


def estimated_target_date(observed: pd.DataFrame, horizon: int) -> pd.Timestamp:
    gap = observed.index.to_series().diff().dt.days.tail(90).dropna().mean()
    return observed.index[-1] + pd.Timedelta(days=max(1, round(gap)) * horizon)


def model_rmse(metrics: pd.DataFrame, target: str, horizon: int, model_name: str) -> float:
    rows = metrics[
        metrics["target"].eq(target)
        & metrics["horizon_reports"].eq(horizon)
        & metrics["display_model"].eq(model_name)
        & metrics["strategy"].str.contains("walk_forward", na=False)
    ]
    return float(rows["rmse"].mean()) if not rows.empty else 0.0


def forecast_chart(target: str, observed: pd.DataFrame, when: pd.Timestamp, prediction: float, rmse: float, adjustment: float = 0) -> go.Figure:
    """Build a history, prediction, and empirical uncertainty chart."""
    history = observed[target].tail(60)
    value = prediction + adjustment
    lower, upper = value - 1.96 * rmse, value + 1.96 * rmse
    if target != "net_hhs_flow":
        lower = max(0, lower)
    colours = {
        "hhs_care": ("#0e7490", "#f97316"),
        "hhs_discharges": ("#7c3aed", "#ec4899"),
        "net_hhs_flow": ("#16a34a", "#eab308"),
    }
    historic, future = colours[target]
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=history.index, y=history, name="Observed", line=dict(color=historic, width=2.5)))
    chart.add_trace(go.Scatter(x=[when], y=[upper], mode="markers", marker=dict(color="rgba(0,0,0,0)"), showlegend=False))
    chart.add_trace(go.Scatter(x=[when], y=[lower], mode="markers", fill="tonexty", fillcolor="rgba(249,115,22,.18)", marker=dict(color="rgba(0,0,0,0)"), name="95% interval"))
    chart.add_trace(go.Scatter(x=[observed.index[-1], when], y=[history.iloc[-1], value], mode="lines+markers", line=dict(color=future, width=3, dash="dash"), name="Forecast"))
    chart.update_layout(title=TARGET_LABELS[target], height=325, margin=dict(l=8, r=8, t=44, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,.75)", legend=dict(orientation="h", y=1.05))
    return chart


def rank_models(metrics: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rank models within each target based on walk-forward MAE."""
    rows = metrics[
        metrics["horizon_reports"].eq(horizon)
        & metrics["strategy"].str.contains("walk_forward", na=False)
    ]
    ranked = rows.groupby(["target", "model_label"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), observations=("observations", "mean")
    )
    ranked["rank"] = ranked.groupby("target")["mae"].rank(method="dense").astype(int)
    return ranked.sort_values(["target", "rank", "mae"])
