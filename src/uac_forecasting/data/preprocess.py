"""Clean and validate the raw UAC operational data.

Run after installing the project:
    python -m uac_forecasting.preprocess

The raw source is never modified.  The pipeline creates two outputs:
* ``uac_observed_clean.csv``: one validated row per reported date.
* ``uac_reporting_calendar.csv``: every calendar date, with unreported days
  explicitly flagged rather than silently imputed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "Data" / "Raw_Data.csv"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
CLEAN_DIR = PROCESSED_DIR / "clean"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
FEATURES_DIR = PROCESSED_DIR / "features"
VALIDATION_DIR = PROCESSED_DIR / "validation"
MODELS_DIR = PROCESSED_DIR / "models"

COLUMN_RENAMES = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "cbp_apprehensions",
    "Children in CBP custody": "cbp_care",
    "Children transferred out of CBP custody": "cbp_transfers_to_hhs",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "hhs_discharges",
}

MEASURE_COLUMNS = list(COLUMN_RENAMES.values())[1:]


def load_and_clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Return validated reported observations in chronological order."""
    raw = pd.read_csv(raw_path, skip_blank_lines=True)
    frame = raw.dropna(how="all").rename(columns=COLUMN_RENAMES).copy()

    unexpected_columns = set(frame.columns).difference(COLUMN_RENAMES.values())
    if unexpected_columns:
        raise ValueError(f"Unexpected source columns: {sorted(unexpected_columns)}")

    frame["date"] = pd.to_datetime(frame["date"], format="%B %d, %Y", errors="raise")
    for column in MEASURE_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace(",", "", regex=False), errors="raise"
        )

    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Missing values found in required columns: {missing}")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate reporting dates found in source data.")
    if (frame[MEASURE_COLUMNS] < 0).any().any():
        raise ValueError("Negative operational counts found in source data.")

    frame = frame.sort_values("date").reset_index(drop=True)
    frame["net_hhs_flow"] = frame["cbp_transfers_to_hhs"] - frame["hhs_discharges"]
    frame["report_day_of_week"] = frame["date"].dt.day_name()
    frame["days_since_previous_report"] = frame["date"].diff().dt.days.fillna(0).astype(int)
    return frame.set_index("date")


def build_reporting_calendar(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Expose reporting gaps without treating them as zero-demand days."""
    calendar = pd.DataFrame(
        {"date": pd.date_range(cleaned.index.min(), cleaned.index.max(), freq="D")}
    )
    reported_dates = cleaned.index.to_frame(index=False, name="date")
    calendar = calendar.merge(reported_dates, on="date", how="left", indicator=True)
    calendar["is_reported"] = calendar.pop("_merge").eq("both")
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    return calendar


def run_preprocessing() -> tuple[Path, Path]:
    """Create analysis-ready files and return their paths."""
    cleaned = load_and_clean()
    calendar = build_reporting_calendar(cleaned)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = CLEAN_DIR / "uac_observed_clean.csv"
    calendar_path = CLEAN_DIR / "uac_reporting_calendar.csv"
    cleaned.to_csv(clean_path, index=True, date_format="%Y-%m-%d")
    calendar.to_csv(calendar_path, index=False, date_format="%Y-%m-%d")

    print(f"Wrote {len(cleaned)} validated observations to {clean_path}")
    print(f"Wrote {len(calendar)} calendar days to {calendar_path}")
    print(f"Unreported calendar days: {(~calendar['is_reported']).sum()}")
    return clean_path, calendar_path


if __name__ == "__main__":
    run_preprocessing()
