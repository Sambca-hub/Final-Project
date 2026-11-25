"""Dataset upload, validation, and session-state management for SmartEdu."""

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = ["Department", "Semester", "Subject", "Marks", "Attendance"]


def read_uploaded_dataset(uploaded_file):
    """Read a user-uploaded CSV or Excel dataset."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def validate_dataset(df):
    """Raise ValueError when the dataset is empty or lacks required columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")


def render_dataset_uploader():
    """Render the sidebar uploader and return the active dataset and filename."""
    st.sidebar.subheader("📂 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a student dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file containing student academic records.",
        key="student_dataset_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_df = read_uploaded_dataset(uploaded_file)
            validate_dataset(uploaded_df)
        except Exception as exc:
            st.sidebar.error(f"Could not read file: {exc}")
            st.session_state.pop("student_dataset", None)
            st.session_state.pop("dataset_name", None)
        else:
            st.session_state["student_dataset"] = uploaded_df
            st.session_state["dataset_name"] = uploaded_file.name
            st.sidebar.success(f"Loaded {len(uploaded_df)} student records.")

    return (
        st.session_state.get("student_dataset"),
        st.session_state.get("dataset_name"),
    )


def require_dataset(df):
    """Display a helpful message and return False when no dataset is loaded."""
    if df is None:
        st.warning("Please upload a student dataset using the uploader in the sidebar.")
        st.info("Supported file formats: CSV (.csv) and Excel (.xlsx, .xls).")
        st.markdown(
            "**Required columns:** `Department`, `Semester`, `Subject`, `Marks`, `Attendance`."
        )
        return False
    return True
