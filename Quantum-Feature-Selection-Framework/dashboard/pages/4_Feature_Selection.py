"""
4_Feature_Selection.py
========================
ANOVA (F-test) based classical feature selection page: choose top-K,
run selection, and visualize the dimensionality reduction achieved.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.feature_selection import select_features_anova
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
    require_stage,
)
from src.utils import DEFAULT_TOP_K_FEATURES, SKeys, dataframe_to_csv_bytes
from src.visualization import plot_feature_reduction, plot_feature_scores

configure_page("Feature Selection", page_icon="🎯")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="🎯 Feature Selection",
    subtitle="Reduce thousands of genes to the most statistically discriminative features using an ANOVA F-test (SelectKBest).",
)

if not require_stage(SKeys.PREPROCESS_DONE, "Preprocessing hasn't been run yet.", "Preprocessing"):
    render_footer()
    st.stop()

X_train = st.session_state[SKeys.X_TRAIN]
X_test = st.session_state[SKeys.X_TEST]
y_train = st.session_state[SKeys.Y_TRAIN]
feature_names = st.session_state[SKeys.FEATURE_NAMES]
n_total_features = X_train.shape[1]

render_section_title("⚙️", "Feature Selection Configuration")
top_k = st.slider(
    "Number of top features to select (K)",
    min_value=4,
    max_value=n_total_features,
    value=min(DEFAULT_TOP_K_FEATURES, n_total_features),
    step=1,
    help="ANOVA F-test ranks every gene by how well it separates the classes; the top K are kept.",
)

run = st.button("▶️ Run ANOVA Feature Selection", type="primary")

if run:
    with st.spinner("Scoring features with ANOVA F-test..."):
        try:
            fs_result = select_features_anova(X_train, X_test, y_train, feature_names, k=top_k)
        except Exception as exc:  # noqa: BLE001
            render_callout(f"❌ Feature selection failed: {exc}", kind="warning")
            st.stop()

    st.session_state[SKeys.SELECTOR] = fs_result["selector"]
    st.session_state[SKeys.X_TRAIN_SELECTED] = fs_result["X_train_selected"]
    st.session_state[SKeys.X_TEST_SELECTED] = fs_result["X_test_selected"]
    st.session_state[SKeys.SELECTED_FEATURE_NAMES] = fs_result["selected_feature_names"]
    st.session_state[SKeys.FEATURE_SCORES] = fs_result["feature_scores"]
    st.session_state[SKeys.TOP_K] = top_k
    st.session_state[SKeys.FEATURE_SELECTION_DONE] = True
    st.session_state[SKeys.QUANTUM_ANALYSIS_DONE] = False
    st.session_state[SKeys.CLASSIFICATION_DONE] = False
    st.success(f"✅ Selected the top {fs_result['n_selected_features']} features out of {fs_result['n_original_features']}.")

if not require_stage(SKeys.FEATURE_SELECTION_DONE, "Feature selection hasn't been run yet.", "Feature Selection (click Run above)"):
    render_footer()
    st.stop()

selected_feature_names = st.session_state[SKeys.SELECTED_FEATURE_NAMES]
feature_scores = st.session_state[SKeys.FEATURE_SCORES]
n_selected = len(selected_feature_names)
reduction_pct = round((1 - n_selected / n_total_features) * 100, 2)

render_section_title("📊", "Selection Summary")
m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Original Features", f"{n_total_features:,}", "genes before selection")
with m2:
    render_metric_card("Selected Features", f"{n_selected:,}", "genes retained", variant="success")
with m3:
    render_metric_card("Reduction", f"{reduction_pct}%", "dimensionality reduced", variant="accent")
with m4:
    render_metric_card("Training Shape", f"{st.session_state[SKeys.X_TRAIN_SELECTED].shape}", "samples × features", variant="secondary")

col1, col2 = st.columns(2)
with col1:
    render_section_title("📉", "Original vs Selected Feature Count")
    st.plotly_chart(plot_feature_reduction(n_total_features, n_selected), use_container_width=True)
with col2:
    render_section_title("🏅", "Top Feature Scores")
    st.plotly_chart(plot_feature_scores(feature_scores, top_n=min(25, n_selected)), use_container_width=True)

render_section_title("📋", "Selected Feature Details")
st.dataframe(feature_scores[feature_scores["selected"]].reset_index(drop=True), use_container_width=True, height=350)

st.download_button(
    "📥 Download Selected Features (CSV)",
    data=dataframe_to_csv_bytes(feature_scores[feature_scores["selected"]]),
    file_name="selected_features.csv",
    mime="text/csv",
)

st.divider()
next_col1, _ = st.columns([1, 3])
with next_col1:
    if st.button("➡️ Proceed to Quantum Analysis", type="primary", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/5_Quantum_Analysis.py")

render_footer()
