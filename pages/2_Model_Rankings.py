"""Model-comparison page."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uac_forecasting.dashboard import TARGET_LABELS, apply_dashboard_style, load_metrics, rank_models  # noqa: E402


st.set_page_config(page_title="Model Rankings | UAC", page_icon="🏆", layout="wide")
apply_dashboard_style()
st.title("Model Rankings")

horizon = st.selectbox("Forecast horizon", (1, 7, 14), format_func=lambda value: f"{value} reports ahead")
ranking = rank_models(load_metrics(), horizon)

for column, target in zip(st.columns(3), TARGET_LABELS):
    table = ranking[ranking["target"].eq(target)][["rank", "model_label", "mae", "rmse", "observations"]].copy()
    table.columns = ["Rank", "Model", "MAE", "RMSE", "Eval. points"]
    with column:
        st.markdown(f"### {TARGET_LABELS[target]}")
        st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("MAE comparison")
chart_data = ranking.copy()
chart_data["target"] = chart_data["target"].map(TARGET_LABELS)
chart = px.bar(chart_data, x="model_label", y="mae", color="target", barmode="group", labels={"model_label": "Model", "mae": "Walk-forward MAE", "target": "Target"})
chart.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,.75)")
st.plotly_chart(chart, use_container_width=True)
st.caption("Rank 1 has the lowest walk-forward MAE. Some model families use sampled folds, so use rankings as guidance rather than a final deployment decision.")
