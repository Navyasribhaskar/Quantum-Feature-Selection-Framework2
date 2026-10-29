"""
7_Results.py
=============
Comprehensive results dashboard: accuracy cards, confusion matrices,
classification reports, ROC curves, the feature-selection graph,
classical-vs-quantum comparison charts, and an auto-generated
performance summary.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.classifier import build_comparison_table
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
from src.utils import SKeys
from src.visualization import (
    plot_accuracy_comparison,
    plot_classification_report_heatmap,
    plot_confusion_matrix,
    plot_feature_reduction,
    plot_metric_comparison_grouped,
    plot_pca_scatter,
    plot_roc_curves,
)

configure_page("Results", page_icon="📊")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="📊 Results",
    subtitle="A complete, side-by-side comparison of the classical and quantum-enhanced classification pipelines.",
)

if not require_stage(SKeys.CLASSIFICATION_DONE, "Classification hasn't been run yet.", "Classification"):
    render_footer()
    st.stop()

classical_metrics = st.session_state[SKeys.CLASSICAL_METRICS]
quantum_metrics = st.session_state[SKeys.QUANTUM_METRICS]
target_names = classical_metrics["target_names"]
y_test = st.session_state[SKeys.Y_TEST]

# --------------------------------------------------------------------------
# Accuracy cards
# --------------------------------------------------------------------------

render_section_title("🏆", "Headline Accuracy")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card("Classical SVM Accuracy", f"{classical_metrics['accuracy']:.2%}")
with c2:
    render_metric_card("Quantum SVM Accuracy", f"{quantum_metrics['accuracy']:.2%}", variant="secondary")
with c3:
    diff = classical_metrics["accuracy"] - quantum_metrics["accuracy"]
    leader = "Classical" if diff > 0 else ("Quantum" if diff < 0 else "Tie")
    render_metric_card("Accuracy Gap", f"{abs(diff):.2%}", f"{leader} leads" if diff != 0 else "Tie", variant="accent")
with c4:
    _n_before = st.session_state[SKeys.X_TRAIN].shape[1]
    _n_after = st.session_state[SKeys.X_TRAIN_SELECTED].shape[1]
    _reduction_pct = round((1 - _n_after / _n_before) * 100, 1) if _n_before else 0.0
    render_metric_card(
        "Feature Reduction",
        f"{_reduction_pct}%",
        "genes reduced via ANOVA",
        variant="success",
    )

# --------------------------------------------------------------------------
# Confusion matrices
# --------------------------------------------------------------------------

render_section_title("🧩", "Confusion Matrices")
cm_col1, cm_col2 = st.columns(2)
with cm_col1:
    st.plotly_chart(
        plot_confusion_matrix(classical_metrics["confusion_matrix"], target_names, "Classical SVM"),
        use_container_width=True,
    )
with cm_col2:
    st.plotly_chart(
        plot_confusion_matrix(quantum_metrics["confusion_matrix"], target_names, "Quantum-Enhanced SVM"),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Classification reports
# --------------------------------------------------------------------------

render_section_title("📄", "Classification Report Heatmaps")
rep_col1, rep_col2 = st.columns(2)
with rep_col1:
    st.plotly_chart(
        plot_classification_report_heatmap(classical_metrics["classification_report"], "Classical SVM"),
        use_container_width=True,
    )
with rep_col2:
    st.plotly_chart(
        plot_classification_report_heatmap(quantum_metrics["classification_report"], "Quantum-Enhanced SVM"),
        use_container_width=True,
    )

with st.expander("📃 View raw classification report text"):
    t1, t2 = st.tabs(["Classical", "Quantum"])
    with t1:
        st.code(classical_metrics["classification_report_text"], language=None)
    with t2:
        st.code(quantum_metrics["classification_report_text"], language=None)

# --------------------------------------------------------------------------
# ROC curves
# --------------------------------------------------------------------------

render_section_title("📈", "ROC Curves (One-vs-Rest)")
roc_col1, roc_col2 = st.columns(2)
with roc_col1:
    fig = plot_roc_curves(y_test, classical_metrics["y_proba"], target_names, "Classical SVM ROC")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        render_callout("ROC curve unavailable for the Classical SVM (no probability estimates).", kind="info")
with roc_col2:
    fig = plot_roc_curves(y_test, quantum_metrics["y_proba"], target_names, "Quantum-Enhanced SVM ROC")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        render_callout("ROC curve unavailable for the Quantum SVM (no probability estimates).", kind="info")

# --------------------------------------------------------------------------
# Feature selection recap
# --------------------------------------------------------------------------

render_section_title("🎯", "Feature Selection Recap")
n_original = st.session_state[SKeys.X_TRAIN].shape[1]
n_selected = st.session_state[SKeys.X_TRAIN_SELECTED].shape[1]
st.plotly_chart(plot_feature_reduction(n_original, n_selected), use_container_width=True)

# --------------------------------------------------------------------------
# Classical vs Quantum comparison
# --------------------------------------------------------------------------

render_section_title("⚖️", "Classical vs. Quantum: Comparison Charts")
comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    st.plotly_chart(
        plot_accuracy_comparison(classical_metrics["accuracy"], quantum_metrics["accuracy"]),
        use_container_width=True,
    )
with comp_col2:
    comparison = build_comparison_table(classical_metrics, quantum_metrics)
    st.plotly_chart(plot_metric_comparison_grouped(comparison), use_container_width=True)

render_section_title("🗺️", "Feature Space Visualization (PCA Projection)")
pca_col1, pca_col2 = st.columns(2)
with pca_col1:
    st.plotly_chart(
        plot_pca_scatter(
            st.session_state[SKeys.X_TEST_SELECTED], y_test, target_names, "Classical Feature Space (Test Set)"
        ),
        use_container_width=True,
    )
with pca_col2:
    st.plotly_chart(
        plot_pca_scatter(
            st.session_state[SKeys.X_TEST_QUANTUM_FEATURES], y_test, target_names, "Quantum Feature Space (Test Set)"
        ),
        use_container_width=True,
    )

# --------------------------------------------------------------------------
# Performance summary
# --------------------------------------------------------------------------

render_section_title("📝", "Performance Summary")
better = "Classical SVM" if classical_metrics["accuracy"] >= quantum_metrics["accuracy"] else "Quantum-Enhanced SVM"
gap_pp = abs(classical_metrics["accuracy"] - quantum_metrics["accuracy"]) * 100

summary_text = f"""
On this dataset, the **{better}** achieved the higher test accuracy
(**{max(classical_metrics['accuracy'], quantum_metrics['accuracy']):.2%}** vs.
**{min(classical_metrics['accuracy'], quantum_metrics['accuracy']):.2%}**), a gap of
**{gap_pp:.1f} percentage points**. Classical feature selection reduced the dataset from
**{n_original:,}** genes to the top **{n_selected}** using an ANOVA F-test, of which the
first **{st.session_state.get(SKeys.N_QUBITS, 4)}** were angle-encoded into a quantum
circuit executed with **{st.session_state.get(SKeys.SHOTS, 256)}** shots per sample.

This head-to-head comparison is the core deliverable of the framework: it does not assume
quantum superiority, but instead provides a reproducible, side-by-side evaluation of a
classical ANOVA + SVM baseline against a quantum angle-encoding + SVM pipeline on the same
train/test split.
"""
render_callout(summary_text, kind="info")

render_footer()
