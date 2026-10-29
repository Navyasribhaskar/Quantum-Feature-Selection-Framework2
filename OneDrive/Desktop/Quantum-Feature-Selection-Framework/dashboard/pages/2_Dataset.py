"""
2_Dataset.py
============
Dataset ingestion page: upload a CSV or use the bundled Golub leukemia
dataset, preview it, configure target/metadata columns, and inspect
shape, missing values, class distribution, and summary statistics.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

from src.data_loader import (
    DatasetValidationError,
    get_dataset_info,
    infer_metadata_columns,
    load_default_dataset,
    load_uploaded_dataset,
    suggest_target_column,
    validate_target_column,
)
from src.ui_components import (
    configure_page,
    init_session_state,
    render_callout,
    render_footer,
    render_hero,
    render_metric_card,
    render_pipeline_status,
    render_section_title,
    render_sidebar_brand,
)
from src.utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METADATA_COLUMNS,
    SKeys,
)
from src.visualization import plot_class_distribution, plot_dtype_breakdown, plot_missing_values

configure_page("Dataset", page_icon="📂")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="📂 Dataset",
    subtitle="Upload a compatible gene-expression CSV, or explore the bundled Golub et al. leukemia dataset.",
)

# --------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------

render_section_title("📥", "Load a Dataset")

source = st.radio(
    "Choose a data source",
    options=["Use bundled Golub leukemia dataset", "Upload my own CSV"],
    horizontal=True,
)

df: pd.DataFrame | None = None
dataset_name: str | None = None
load_error: str | None = None

if source == "Use bundled Golub leukemia dataset":
    if st.button("📊 Load Bundled Dataset", type="primary"):
        try:
            df = load_default_dataset(DEFAULT_DATASET_PATH)
            dataset_name = "golub.csv (bundled)"
        except FileNotFoundError as exc:
            load_error = str(exc)
    elif st.session_state.get(SKeys.RAW_DF) is not None and st.session_state.get(SKeys.DATASET_NAME, "").endswith("(bundled)"):
        df = st.session_state[SKeys.RAW_DF]
        dataset_name = st.session_state[SKeys.DATASET_NAME]
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV file (samples as rows, genes/features as columns, one target/label column)",
        type=["csv"],
    )
    if uploaded_file is not None:
        try:
            df = load_uploaded_dataset(uploaded_file)
            dataset_name = uploaded_file.name
        except DatasetValidationError as exc:
            load_error = str(exc)
    elif st.session_state.get(SKeys.RAW_DF) is not None and not str(st.session_state.get(SKeys.DATASET_NAME, "")).endswith("(bundled)"):
        df = st.session_state[SKeys.RAW_DF]
        dataset_name = st.session_state[SKeys.DATASET_NAME]

if load_error:
    render_callout(f"❌ <b>Could not load dataset:</b> {load_error}", kind="warning")

if df is not None:
    st.session_state[SKeys.RAW_DF] = df
    st.session_state[SKeys.DATASET_NAME] = dataset_name
    # Reset downstream pipeline state whenever a new dataset is loaded.
    for flag in (SKeys.PREPROCESS_DONE, SKeys.FEATURE_SELECTION_DONE, SKeys.QUANTUM_ANALYSIS_DONE, SKeys.CLASSIFICATION_DONE):
        st.session_state.setdefault(flag, False)

# --------------------------------------------------------------------------
# Configuration + preview (only if a dataset is loaded)
# --------------------------------------------------------------------------

if st.session_state.get(SKeys.RAW_DF) is None:
    render_callout(
        "ℹ️ No dataset loaded yet. Load the bundled dataset or upload your own CSV above to get started.",
        kind="info",
    )
    render_footer()
    st.stop()

df = st.session_state[SKeys.RAW_DF]
dataset_name = st.session_state[SKeys.DATASET_NAME]

st.success(f"✅ Dataset loaded: **{dataset_name}** — {df.shape[0]} rows × {df.shape[1]} columns")

render_section_title("⚙️", "Dataset Configuration")
col1, col2 = st.columns(2)

with col1:
    suggested_target = st.session_state.get(SKeys.TARGET_COLUMN) or suggest_target_column(df)
    target_column = st.selectbox(
        "Target / label column",
        options=list(df.columns),
        index=list(df.columns).index(suggested_target) if suggested_target in df.columns else 0,
        help="The column containing the class labels to predict (e.g. cancer subtype).",
    )

with col2:
    default_meta = st.session_state.get(SKeys.METADATA_COLUMNS)
    if default_meta is None:
        default_meta = infer_metadata_columns(df, DEFAULT_METADATA_COLUMNS)
    metadata_columns = st.multiselect(
        "Metadata columns to exclude (not gene-expression features)",
        options=[c for c in df.columns if c != target_column],
        default=[c for c in default_meta if c != target_column],
        help="Clinical/administrative columns (e.g. patient ID, gender) that should be dropped before modelling.",
    )

try:
    validate_target_column(df, target_column)
    st.session_state[SKeys.TARGET_COLUMN] = target_column
    st.session_state[SKeys.METADATA_COLUMNS] = metadata_columns
except DatasetValidationError as exc:
    render_callout(f"❌ {exc}", kind="warning")
    render_footer()
    st.stop()

# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------

render_section_title("👁️", "Dataset Preview")
preview_rows = st.slider("Rows to preview", min_value=5, max_value=min(50, df.shape[0]), value=min(10, df.shape[0]))
st.dataframe(df.head(preview_rows), use_container_width=True)

# --------------------------------------------------------------------------
# Dataset information
# --------------------------------------------------------------------------

render_section_title("📊", "Dataset Information")
info = get_dataset_info(df, target_column=target_column)
st.session_state[SKeys.DATASET_INFO] = info

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Samples", f"{info.n_rows:,}", "rows")
with m2:
    render_metric_card("Total Columns", f"{info.n_columns:,}", f"{info.n_numeric_columns} numeric", variant="accent")
with m3:
    render_metric_card("Missing Values", f"{info.missing_values_total:,}", "total cells", variant="warning" if info.missing_values_total else "success")
with m4:
    render_metric_card("Memory Usage", f"{info.memory_usage_mb} MB", "in-memory size", variant="secondary")

col_a, col_b = st.columns(2)
with col_a:
    render_section_title("🧬", "Class Distribution")
    if info.class_distribution:
        st.plotly_chart(plot_class_distribution(info.class_distribution), use_container_width=True)
    else:
        render_callout("Target column has no class distribution to display.", kind="info")
with col_b:
    render_section_title("🧩", "Column Data Types")
    st.plotly_chart(plot_dtype_breakdown(info.n_numeric_columns, info.n_non_numeric_columns), use_container_width=True)

render_section_title("🚨", "Missing Values by Column")
st.plotly_chart(plot_missing_values(info.missing_by_column), use_container_width=True)

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

render_section_title("📈", "Summary Statistics")
numeric_df = df.select_dtypes(include="number")
if not numeric_df.empty:
    st.dataframe(numeric_df.describe().T, use_container_width=True)
else:
    render_callout("No numeric columns available for summary statistics.", kind="info")

render_section_title("🗂️", "Column Overview")
dtype_table = pd.DataFrame(
    {
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Missing": [int(df[c].isnull().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }
)
st.dataframe(dtype_table, use_container_width=True, height=320)

st.divider()
next_col1, _ = st.columns([1, 3])
with next_col1:
    if st.button("➡️ Proceed to Preprocessing", type="primary", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/3_Preprocessing.py")

render_footer()
