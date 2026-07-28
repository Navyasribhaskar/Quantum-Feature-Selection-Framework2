"""
8_Download.py
===============
Export page: download predictions, metrics, classification reports,
selected features, key graphs (PNG), and a consolidated PDF report.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

import pandas as pd
import streamlit as st

from src.classifier import build_comparison_table
from src.ui_components import (
    configure_page,
    init_session_state,
    render_callout,
    render_footer,
    render_hero,
    render_pipeline_status,
    render_section_title,
    render_sidebar_brand,
    require_stage,
)
from src.utils import SKeys, dataframe_to_csv_bytes
from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_feature_reduction,
)

configure_page("Download", page_icon="📥")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="📥 Download Center",
    subtitle="Export predictions, metrics, reports, selected features, graphs, and a consolidated PDF summary.",
)

if not require_stage(SKeys.CLASSIFICATION_DONE, "Classification hasn't been run yet.", "Classification"):
    render_footer()
    st.stop()

label_encoder = st.session_state[SKeys.LABEL_ENCODER]
y_test = st.session_state[SKeys.Y_TEST]
classical_metrics = st.session_state[SKeys.CLASSICAL_METRICS]
quantum_metrics = st.session_state[SKeys.QUANTUM_METRICS]
target_names = classical_metrics["target_names"]

# --------------------------------------------------------------------------
# 1. Predictions CSV
# --------------------------------------------------------------------------

render_section_title("🔮", "Predictions")
pred_df = pd.DataFrame(
    {
        "sample_index": range(len(y_test)),
        "true_label": label_encoder.inverse_transform(y_test),
        "classical_svm_prediction": label_encoder.inverse_transform(classical_metrics["y_pred"]),
        "quantum_svm_prediction": label_encoder.inverse_transform(quantum_metrics["y_pred"]),
    }
)
pred_df["classical_correct"] = pred_df["true_label"] == pred_df["classical_svm_prediction"]
pred_df["quantum_correct"] = pred_df["true_label"] == pred_df["quantum_svm_prediction"]
st.dataframe(pred_df, use_container_width=True, height=250)
st.download_button(
    "📥 Download Predictions CSV",
    data=dataframe_to_csv_bytes(pred_df),
    file_name="predictions.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()

# --------------------------------------------------------------------------
# 2. Metrics CSV
# --------------------------------------------------------------------------

render_section_title("📊", "Metrics")
comparison = build_comparison_table(classical_metrics, quantum_metrics)
metrics_df = pd.DataFrame(comparison)
st.dataframe(metrics_df, use_container_width=True)
st.download_button(
    "📥 Download Metrics CSV",
    data=dataframe_to_csv_bytes(metrics_df),
    file_name="metrics_comparison.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()

# --------------------------------------------------------------------------
# 3. Classification reports
# --------------------------------------------------------------------------

render_section_title("📄", "Classification Reports")
rep_col1, rep_col2 = st.columns(2)
with rep_col1:
    st.download_button(
        "📥 Download Classical Report (TXT)",
        data=classical_metrics["classification_report_text"].encode("utf-8"),
        file_name="classical_svm_classification_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    classical_report_csv = pd.DataFrame(classical_metrics["classification_report"]).T
    st.download_button(
        "📥 Download Classical Report (CSV)",
        data=dataframe_to_csv_bytes(classical_report_csv, index=True),
        file_name="classical_svm_classification_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
with rep_col2:
    st.download_button(
        "📥 Download Quantum Report (TXT)",
        data=quantum_metrics["classification_report_text"].encode("utf-8"),
        file_name="quantum_svm_classification_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    quantum_report_csv = pd.DataFrame(quantum_metrics["classification_report"]).T
    st.download_button(
        "📥 Download Quantum Report (CSV)",
        data=dataframe_to_csv_bytes(quantum_report_csv, index=True),
        file_name="quantum_svm_classification_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# --------------------------------------------------------------------------
# 4. Selected features
# --------------------------------------------------------------------------

render_section_title("🎯", "Selected Features")
if st.session_state.get(SKeys.FEATURE_SCORES) is not None:
    feature_scores = st.session_state[SKeys.FEATURE_SCORES]
    selected_only = feature_scores[feature_scores["selected"]].reset_index(drop=True)
    st.dataframe(selected_only, use_container_width=True, height=250)
    st.download_button(
        "📥 Download Selected Features CSV",
        data=dataframe_to_csv_bytes(selected_only),
        file_name="selected_features.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# --------------------------------------------------------------------------
# 5. Graphs (PNG)
# --------------------------------------------------------------------------

render_section_title("🖼️", "Graphs")
render_callout(
    "PNG export requires the optional <code>kaleido</code> package (listed in requirements.txt). "
    "If a download fails in your environment, the interactive charts are always available on the "
    "Results page.",
    kind="info",
)

graphs = {
    "accuracy_comparison.png": plot_accuracy_comparison(classical_metrics["accuracy"], quantum_metrics["accuracy"]),
    "confusion_matrix_classical.png": plot_confusion_matrix(
        classical_metrics["confusion_matrix"], target_names, "Classical SVM Confusion Matrix"
    ),
    "confusion_matrix_quantum.png": plot_confusion_matrix(
        quantum_metrics["confusion_matrix"], target_names, "Quantum SVM Confusion Matrix"
    ),
}
if st.session_state.get(SKeys.X_TRAIN_SELECTED) is not None:
    graphs["feature_reduction.png"] = plot_feature_reduction(
        st.session_state[SKeys.X_TRAIN].shape[1], st.session_state[SKeys.X_TRAIN_SELECTED].shape[1]
    )

graph_cols = st.columns(len(graphs))
for col, (filename, fig) in zip(graph_cols, graphs.items()):
    with col:
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                f"📥 {filename}",
                data=png_bytes,
                file_name=filename,
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption(f"⚠️ {filename} export unavailable (kaleido not installed)")

st.divider()

# --------------------------------------------------------------------------
# 6. PDF Report
# --------------------------------------------------------------------------

render_section_title("📕", "Consolidated PDF Report")
render_callout(
    "Generates a single professional PDF summarizing the dataset, preprocessing, feature "
    "selection, quantum configuration, and full classical-vs-quantum comparison.",
    kind="info",
)

if st.button("📄 Generate PDF Report", type="primary"):
    try:
        from src.report_generator import generate_pdf_report

        context = {
            "dataset_name": st.session_state.get(SKeys.DATASET_NAME),
            "dataset_shape": st.session_state[SKeys.RAW_DF].shape if st.session_state.get(SKeys.RAW_DF) is not None else None,
            "target_column": st.session_state.get(SKeys.TARGET_COLUMN),
            "dropped_metadata_columns": st.session_state.get("preprocess_dropped_cols", []),
            "class_mapping": st.session_state.get("preprocess_class_mapping", {}),
            "n_original_features": st.session_state[SKeys.X_TRAIN].shape[1],
            "n_selected_features": st.session_state[SKeys.X_TRAIN_SELECTED].shape[1],
            "reduction_percentage": round(
                (1 - st.session_state[SKeys.X_TRAIN_SELECTED].shape[1] / st.session_state[SKeys.X_TRAIN].shape[1]) * 100,
                2,
            ),
            "n_qubits": st.session_state.get(SKeys.N_QUBITS),
            "shots": st.session_state.get(SKeys.SHOTS),
            "classical_metrics": classical_metrics,
            "quantum_metrics": quantum_metrics,
        }
        pdf_bytes = generate_pdf_report(context)
        st.session_state["_pdf_report_bytes"] = pdf_bytes
        st.success("✅ PDF report generated successfully!")
    except Exception as exc:  # noqa: BLE001
        render_callout(f"❌ Could not generate PDF report: {exc}", kind="warning")

if st.session_state.get("_pdf_report_bytes"):
    st.download_button(
        "📥 Download PDF Report",
        data=st.session_state["_pdf_report_bytes"],
        file_name="quantum_feature_selection_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

render_footer()
