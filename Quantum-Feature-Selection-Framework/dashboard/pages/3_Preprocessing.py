"""
3_Preprocessing.py
===================
Runs and visualizes the data cleaning / encoding / scaling / splitting
pipeline (src/preprocessing.py) with a step-by-step progress indicator
and interactive before/after tables.
"""

from __future__ import annotations

import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

from src.preprocessing import run_preprocessing_pipeline
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
    render_step_pills,
    require_stage,
)
from src.utils import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, SKeys

configure_page("Preprocessing", page_icon="🧹")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="🧹 Preprocessing",
    subtitle="Clean metadata, handle missing values, encode labels, standardize features, and split into train/test sets.",
)

if st.session_state.get(SKeys.RAW_DF) is None:
    render_callout("⚠️ No dataset loaded. Please load a dataset on the <b>Dataset</b> page first.", kind="warning")
    render_footer()
    st.stop()

df = st.session_state[SKeys.RAW_DF]
target_column = st.session_state[SKeys.TARGET_COLUMN]
metadata_columns = st.session_state[SKeys.METADATA_COLUMNS] or []

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

render_section_title("⚙️", "Preprocessing Configuration")
c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Test set size", min_value=0.1, max_value=0.4, value=DEFAULT_TEST_SIZE, step=0.05)
with c2:
    missing_strategy = st.selectbox("Missing value strategy", options=["zero", "mean", "median"], index=0)
with c3:
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=DEFAULT_RANDOM_STATE, step=1)

run = st.button("▶️ Run Preprocessing Pipeline", type="primary")

STEP_LABELS = [
    "Remove Metadata Columns",
    "Separate Features and Target",
    "Handle Missing Values",
    "Label Encoding",
    "Feature Standardization",
    "Train/Test Split",
]

if run:
    progress_bar = st.progress(0, text="Starting preprocessing pipeline...")
    for i, label in enumerate(STEP_LABELS):
        progress_bar.progress((i + 1) / len(STEP_LABELS), text=f"Running: {label}...")
        time.sleep(0.15)  # brief pause so the progress indicator is visibly informative

    try:
        result = run_preprocessing_pipeline(
            df=df,
            target_column=target_column,
            metadata_columns=metadata_columns,
            test_size=test_size,
            random_state=int(random_state),
            missing_value_strategy=missing_strategy,
        )
    except Exception as exc:  # noqa: BLE001
        progress_bar.empty()
        render_callout(f"❌ Preprocessing failed: {exc}", kind="warning")
        st.stop()

    progress_bar.progress(1.0, text="Preprocessing complete!")
    time.sleep(0.2)
    progress_bar.empty()

    st.session_state[SKeys.X_TRAIN] = result["X_train"]
    st.session_state[SKeys.X_TEST] = result["X_test"]
    st.session_state[SKeys.Y_TRAIN] = result["y_train"]
    st.session_state[SKeys.Y_TEST] = result["y_test"]
    st.session_state[SKeys.FEATURE_NAMES] = result["feature_names"]
    st.session_state[SKeys.LABEL_ENCODER] = result["label_encoder"]
    st.session_state[SKeys.SCALER] = result["scaler"]
    st.session_state[SKeys.PREPROCESS_LOG] = result["steps"]
    st.session_state["preprocess_class_mapping"] = result["class_mapping"]
    st.session_state["preprocess_dropped_cols"] = result["dropped_metadata_columns"]
    st.session_state["preprocess_missing_counts"] = result["missing_value_counts"]
    st.session_state[SKeys.PREPROCESS_DONE] = True
    # Downstream stages are invalidated whenever preprocessing re-runs.
    st.session_state[SKeys.FEATURE_SELECTION_DONE] = False
    st.session_state[SKeys.QUANTUM_ANALYSIS_DONE] = False
    st.session_state[SKeys.CLASSIFICATION_DONE] = False

    st.success("✅ Preprocessing completed successfully!")

# --------------------------------------------------------------------------
# Results (persist across reruns once computed)
# --------------------------------------------------------------------------

if not require_stage(SKeys.PREPROCESS_DONE, "The preprocessing pipeline hasn't been run yet.", "Preprocessing (click Run above)"):
    render_footer()
    st.stop()

render_section_title("✅", "Transformation Steps")
render_step_pills(STEP_LABELS, done_upto=len(STEP_LABELS) - 1)

with st.container():
    for entry in st.session_state[SKeys.PREPROCESS_LOG]:
        st.markdown(f"**{entry.timestamp} — {entry.step}**  \n{entry.detail}")

render_section_title("📊", "Resulting Shapes")
m1, m2, m3, m4 = st.columns(4)
X_train = st.session_state[SKeys.X_TRAIN]
X_test = st.session_state[SKeys.X_TEST]
with m1:
    render_metric_card("Training Samples", f"{X_train.shape[0]}", f"{X_train.shape[1]} features")
with m2:
    render_metric_card("Testing Samples", f"{X_test.shape[0]}", f"{X_test.shape[1]} features", variant="secondary")
with m3:
    render_metric_card("Dropped Metadata Cols", f"{len(st.session_state['preprocess_dropped_cols'])}", variant="accent")
with m4:
    render_metric_card("Missing Values Filled", f"{sum(st.session_state['preprocess_missing_counts'].values())}", variant="warning")

render_section_title("🏷️", "Label Encoding Map")
class_mapping = st.session_state["preprocess_class_mapping"]
mapping_df = pd.DataFrame(
    {"Encoded Value": list(class_mapping.keys()), "Original Class Label": list(class_mapping.values())}
)
st.dataframe(mapping_df, use_container_width=True)

render_section_title("📐", "Feature Standardization: Before vs. After (Training Set)")
feature_names = st.session_state[SKeys.FEATURE_NAMES]
n_preview = min(10, len(feature_names))
raw_preview = df[feature_names[:n_preview]].head(5).reset_index(drop=True)
scaled_preview = pd.DataFrame(X_train[:5, :n_preview], columns=feature_names[:n_preview])

tab1, tab2 = st.tabs(["🔴 Before Scaling (raw values)", "🟢 After Scaling (standardized)"])
with tab1:
    st.dataframe(raw_preview, use_container_width=True)
with tab2:
    st.dataframe(scaled_preview.round(4), use_container_width=True)

st.divider()
next_col1, _ = st.columns([1, 3])
with next_col1:
    if st.button("➡️ Proceed to Feature Selection", type="primary", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/4_Feature_Selection.py")

render_footer()
