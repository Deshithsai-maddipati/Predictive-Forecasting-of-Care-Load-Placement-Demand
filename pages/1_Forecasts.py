"""Forecast and capacity-scenario page."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uac_forecasting.dashboard import (  # noqa: E402
    MODEL_OPTIONS,
    TARGET_LABELS,
    apply_dashboard_style,
    estimated_target_date,
    forecast_chart,
    load_metrics,
    make_forecast,
    model_rmse,
)
from uac_forecasting.data.preprocess import load_and_clean  # noqa: E402

dialog = st.dialog if hasattr(st, "dialog") else st.experimental_dialog

@dialog("Forecast detail")
def show_forecast_modal(
    title: str,
    chart,
    latest_value: float,
    forecast_value: float,
    horizon: int,
    when,
) -> None:
    """Show an expanded interactive chart and its forecast context."""
    st.subheader(title)
    chart.update_layout(height=620, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(chart, use_container_width=True)
    latest, forecast = st.columns(2)
    latest.metric("Latest observed value", f"{latest_value:,.0f}")
    forecast.metric(f"Forecast in {horizon} reports", f"{forecast_value:,.0f}")
    st.caption(f"Estimated target reporting date: {when.date()}")


st.set_page_config(page_title="Forecasts | UAC", page_icon="📈", layout="wide")
apply_dashboard_style()
st.title("Operational Forecasts")

with st.sidebar:
    st.header("Forecast controls")
    horizon = st.selectbox("Forecast horizon", (1, 7, 14), format_func=lambda value: f"{value} reports ahead")
    model_name = st.selectbox("Forecast model", MODEL_OPTIONS)
    scenario = st.slider("Net-flow scenario adjustment per report", -100, 100, 0)

observed = load_and_clean()
metrics = load_metrics()
when = estimated_target_date(observed, horizon)
forecasts = {target: make_forecast(target, horizon, model_name, observed) for target in TARGET_LABELS}
care_scenario = max(0, forecasts["hhs_care"] + scenario * horizon)

care, discharge, flow = st.columns(3)
care.metric("Forecast HHS care", f"{forecasts['hhs_care']:,.0f}", f"Scenario: {care_scenario - forecasts['hhs_care']:+,.0f}")
discharge.metric("Forecast discharge demand", f"{forecasts['hhs_discharges']:,.0f}")
flow.metric("Forecast net HHS flow", f"{forecasts['net_hhs_flow']:,.0f}")
st.caption(f"Estimated target reporting date: {when.date()}. Shaded bands are empirical ±1.96 RMSE intervals.")

for column, target in zip(st.columns(3), TARGET_LABELS):
    adjustment = scenario * horizon if target == "hhs_care" else 0
    displayed_forecast = forecasts[target] + adjustment
    chart = forecast_chart(
        target,
        observed,
        when,
        forecasts[target],
        model_rmse(metrics, target, horizon, model_name),
        adjustment,
    )
    with column:
        st.plotly_chart(chart, use_container_width=True)
        if st.button("Expand chart", key=f"expand_{target}", use_container_width=True):
            show_forecast_modal(
                TARGET_LABELS[target],
                chart,
                observed[target].iloc[-1],
                displayed_forecast,
                horizon,
                when,
            )

st.info("The scenario control changes only the displayed HHS-care planning case; it does not retrain or overwrite a model.")
