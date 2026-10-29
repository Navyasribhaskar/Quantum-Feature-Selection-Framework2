"""
classifier.py
==============
Support Vector Machine training and evaluation, for both the classical
(ANOVA-selected genes) and quantum (encoded feature-vector) pipelines.

Mirrors the notebook's SVC(kernel='linear', random_state=42) usage for
both models, and its accuracy / classification_report / confusion_matrix
evaluation calls.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from src.utils import DEFAULT_RANDOM_STATE, DEFAULT_SVM_KERNEL, get_logger

log = get_logger(__name__)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = DEFAULT_SVM_KERNEL,
    random_state: int = DEFAULT_RANDOM_STATE,
    probability: bool = True,
) -> SVC:
    """Train a Support Vector Classifier.

    `probability=True` enables `predict_proba`, which is needed for the
    Results page's ROC-curve visualization; it does not change the
    decision boundary learned by the notebook's original SVC.
    """
    model = SVC(kernel=kernel, random_state=random_state, probability=probability)
    model.fit(X_train, y_train)
    log.info("Trained SVM (kernel=%s) on data of shape %s", kernel, X_train.shape)
    return model


def evaluate_model(
    model: SVC,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """Evaluate a trained model and package metrics for UI display.

    Returns a dict with:
        y_pred                (np.ndarray)
        y_proba                (np.ndarray | None)
        accuracy               (float)
        precision_macro         (float)
        recall_macro            (float)
        f1_macro                (float)
        classification_report   (dict, from sklearn `output_dict=True`)
        classification_report_text (str, human-readable)
        confusion_matrix         (np.ndarray)
        target_names             (list[str])
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    target_names = list(label_encoder.classes_)

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
        "target_names": target_names,
    }

    log.info(
        "Evaluation complete. Accuracy=%.4f, F1(macro)=%.4f",
        metrics["accuracy"],
        metrics["f1_macro"],
    )
    return metrics


def build_comparison_table(classical_metrics: dict, quantum_metrics: dict) -> dict:
    """Build a side-by-side comparison of classical vs quantum metrics."""
    return {
        "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
        "Classical SVM": [
            classical_metrics["accuracy"],
            classical_metrics["precision_macro"],
            classical_metrics["recall_macro"],
            classical_metrics["f1_macro"],
        ],
        "Quantum-Enhanced SVM": [
            quantum_metrics["accuracy"],
            quantum_metrics["precision_macro"],
            quantum_metrics["recall_macro"],
            quantum_metrics["f1_macro"],
        ],
    }
