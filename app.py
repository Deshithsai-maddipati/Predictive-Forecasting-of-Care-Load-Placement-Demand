"""Landing page for the UAC Care Forecasting Streamlit application."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uac_forecasting.dashboard import apply_dashboard_style  # noqa: E402
from uac_forecasting.data.preprocess import load_and_clean  # noqa: E402


st.set_page_config(page_title="UAC Care Forecasting", layout="wide")
apply_dashboard_style()

observed = load_and_clean()
st.title("UAC Care Forecasting & Capacity Planning")
st.subheader("Predictive intelligence for proactive child-welfare operations")

left, right = st.columns([2, 1])
with left:
    st.write(
        "Use the pages in the sidebar to generate operational forecasts, compare models, "
        "and explore the reporting data behind the analysis."
    )
    st.markdown("### Available pages")
    st.markdown("- **Forecasts**: HHS care, discharge demand, and net-flow outlook with scenarios.")
    st.markdown("- **Model Rankings**: chronological MAE/RMSE comparison for every target.")
    st.markdown("- **Data Explorer**: historical series, reporting coverage, and data preview.")
with right:
    st.metric("Validated reports", f"{len(observed):,}")
    st.metric("Latest report", observed.index[-1].strftime("%d %b %Y"))
    st.metric("HHS care in latest report", f"{observed['hhs_care'].iloc[-1]:,.0f}")

st.info("Forecast horizons are expressed in future reported observations because the raw source has systematic non-reporting dates.")
