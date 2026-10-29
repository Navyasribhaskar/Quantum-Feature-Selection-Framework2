"""
6_Classification.py
=====================
Trains and evaluates both the Classical SVM (on ANOVA-selected genes)
and the Quantum-Enhanced SVM (on quantum-derived feature vectors), and
displays training/testing/prediction status plus accuracy, precision,
recall, and F1-score for each.
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

from src.classifier import evaluate_model, train_svm
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
from src.utils import DEFAULT_SVM_KERNEL, SKeys

configure_page("Classification", page_icon="🧠")
init_session_state()
render_sidebar_brand()
render_pipeline_status()

render_hero(
    title="🧠 Classification",
    subtitle="Train and evaluate a Classical SVM and a Quantum-Enhanced SVM, side by side.",
)

if not require_stage(SKeys.QUANTUM_ANALYSIS_DONE, "Quantum feature extraction hasn't been run yet.", "Quantum Analysis"):
    render_footer()
    st.stop()

render_section_title("⚙️", "Classifier Configuration")
kernel = st.selectbox("SVM kernel", options=["linear", "rbf", "poly", "sigmoid"], index=0)

run = st.button("🚀 Train Both Classifiers", type="primary")

if run:
    label_encoder = st.session_state[SKeys.LABEL_ENCODER]
    y_train = st.session_state[SKeys.Y_TRAIN]
    y_test = st.session_state[SKeys.Y_TEST]

    progress = st.progress(0, text="Training Classical SVM...")
    classical_model = train_svm(st.session_state[SKeys.X_TRAIN_SELECTED], y_train, kernel=kernel)
    progress.progress(0.35, text="Evaluating Classical SVM on test set...")
    classical_metrics = evaluate_model(classical_model, st.session_state[SKeys.X_TEST_SELECTED], y_test, label_encoder)

    progress.progress(0.55, text="Training Quantum-Enhanced SVM...")
    quantum_model = train_svm(st.session_state[SKeys.X_TRAIN_QUANTUM_FEATURES], y_train, kernel=kernel)
    progress.progress(0.85, text="Evaluating Quantum-Enhanced SVM on test set...")
    quantum_metrics = evaluate_model(quantum_model, st.session_state[SKeys.X_TEST_QUANTUM_FEATURES], y_test, label_encoder)

    progress.progress(1.0, text="Done!")
    time.sleep(0.2)
    progress.empty()

    st.session_state[SKeys.CLASSICAL_MODEL] = classical_model
    st.session_state[SKeys.QUANTUM_MODEL] = quantum_model
    st.session_state[SKeys.CLASSICAL_METRICS] = classical_metrics
    st.session_state[SKeys.QUANTUM_METRICS] = quantum_metrics
    st.session_state[SKeys.CLASSIFICATION_DONE] = True

    st.success("✅ Both classifiers trained and evaluated successfully!")

if not require_stage(SKeys.CLASSIFICATION_DONE, "Classification hasn't been run yet.", "Classification (click Train above)"):
    render_footer()
    st.stop()

classical_metrics = st.session_state[SKeys.CLASSICAL_METRICS]
quantum_metrics = st.session_state[SKeys.QUANTUM_METRICS]
target_names = classical_metrics["target_names"]

render_section_title("🏁", "Training & Testing Status")
status_col1, status_col2 = st.columns(2)
with status_col1:
    render_callout(
        f"✅ <b>Classical SVM</b> — trained on {st.session_state[SKeys.X_TRAIN_SELECTED].shape[0]} samples, "
        f"tested on {st.session_state[SKeys.X_TEST_SELECTED].shape[0]} samples.",
        kind="success",
    )
with status_col2:
    render_callout(
        f"✅ <b>Quantum-Enhanced SVM</b> — trained on {st.session_state[SKeys.X_TRAIN_QUANTUM_FEATURES].shape[0]} samples, "
        f"tested on {st.session_state[SKeys.X_TEST_QUANTUM_FEATURES].shape[0]} samples.",
        kind="success",
    )

render_section_title("📊", "Performance Metrics")
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.markdown("#### 🔵 Classical SVM")
    render_metric_card("Accuracy", f"{classical_metrics['accuracy']:.2%}")
    render_metric_card("Precision (macro)", f"{classical_metrics['precision_macro']:.2%}", variant="secondary")
    render_metric_card("Recall (macro)", f"{classical_metrics['recall_macro']:.2%}", variant="accent")
    render_metric_card("F1-score (macro)", f"{classical_metrics['f1_macro']:.2%}", variant="success")
with metric_col2:
    st.markdown("#### 🟣 Quantum-Enhanced SVM")
    render_metric_card("Accuracy", f"{quantum_metrics['accuracy']:.2%}")
    render_metric_card("Precision (macro)", f"{quantum_metrics['precision_macro']:.2%}", variant="secondary")
    render_metric_card("Recall (macro)", f"{quantum_metrics['recall_macro']:.2%}", variant="accent")
    render_metric_card("F1-score (macro)", f"{quantum_metrics['f1_macro']:.2%}", variant="success")

render_section_title("🔮", "Predictions on Test Set")
y_test = st.session_state[SKeys.Y_TEST]
label_encoder = st.session_state[SKeys.LABEL_ENCODER]
pred_df = pd.DataFrame(
    {
        "Sample #": range(len(y_test)),
        "True Label": label_encoder.inverse_transform(y_test),
        "Classical SVM Prediction": label_encoder.inverse_transform(classical_metrics["y_pred"]),
        "Quantum SVM Prediction": label_encoder.inverse_transform(quantum_metrics["y_pred"]),
    }
)
pred_df["Classical Correct"] = pred_df["True Label"] == pred_df["Classical SVM Prediction"]
pred_df["Quantum Correct"] = pred_df["True Label"] == pred_df["Quantum SVM Prediction"]
st.dataframe(pred_df, use_container_width=True, height=350)

render_section_title("📄", "Classification Reports")
tab1, tab2 = st.tabs(["🔵 Classical SVM", "🟣 Quantum-Enhanced SVM"])
with tab1:
    st.code(classical_metrics["classification_report_text"], language=None)
with tab2:
    st.code(quantum_metrics["classification_report_text"], language=None)

st.divider()
next_col1, _ = st.columns([1, 3])
with next_col1:
    if st.button("➡️ Proceed to Results", type="primary", use_container_width=True):
        if hasattr(st, "switch_page"):
            st.switch_page("pages/7_Results.py")

render_footer()
