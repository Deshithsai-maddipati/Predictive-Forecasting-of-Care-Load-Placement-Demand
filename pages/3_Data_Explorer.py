"""Historical reporting-data exploration page."""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uac_forecasting.dashboard import TARGET_LABELS, apply_dashboard_style  # noqa: E402
from uac_forecasting.data.preprocess import load_and_clean  # noqa: E402


st.set_page_config(page_title="Data Explorer | UAC", page_icon="🔎", layout="wide")
apply_dashboard_style()
st.title("Data Explorer")

observed = load_and_clean()
target = st.selectbox("Operational measure", list(TARGET_LABELS), format_func=lambda value: TARGET_LABELS[value])
view = observed[[target]].reset_index()
chart = px.line(view, x="date", y=target, labels={"date": "Reporting date", target: TARGET_LABELS[target]})
chart.update_traces(line_color="#0e7490")
chart.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,.75)")
st.plotly_chart(chart, use_container_width=True)

left, right = st.columns(2)
left.metric("Date range", f"{observed.index.min().date()} — {observed.index.max().date()}")
right.metric("Reported observations", f"{len(observed):,}")
st.subheader("Latest reported records")
st.dataframe(observed.tail(25), use_container_width=True)
