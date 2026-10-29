"""
visualization.py
==================
All chart-building functions used across the dashboard. Interactive
charts use Plotly (for a modern, zoomable, hoverable feel inside
Streamlit); circuit diagrams and Bloch spheres use Qiskit's own
Matplotlib-based visualizers, since those are purpose-built for quantum
objects and are not easily reproduced in Plotly.

Every function returns a figure object (never calls .show()/st.plotly_chart
itself) so pages stay in control of layout and captions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------------------------------
# Shared color palette (scientific, modern: deep blues/purples + teal accent)
# --------------------------------------------------------------------------

COLOR_PRIMARY = "#4F46E5"       # indigo
COLOR_SECONDARY = "#06B6D4"     # cyan / teal
COLOR_ACCENT = "#8B5CF6"        # violet
COLOR_SUCCESS = "#10B981"       # emerald
COLOR_WARNING = "#F59E0B"       # amber
COLOR_DANGER = "#EF4444"        # red
COLOR_MUTED = "#94A3B8"         # slate

CLASSICAL_COLOR = "#4F46E5"
QUANTUM_COLOR = "#06B6D4"

PLOTLY_TEMPLATE = "plotly_white"

CATEGORICAL_PALETTE = px.colors.qualitative.Bold


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18)),
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=40, r=30, t=60, b=40),
        font=dict(family="Inter, -apple-system, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


# --------------------------------------------------------------------------
# Dataset page
# --------------------------------------------------------------------------


def plot_class_distribution(class_distribution: dict[str, int], title: str = "Class Distribution") -> go.Figure:
    """Bar chart of target-class counts."""
    labels = list(class_distribution.keys())
    values = list(class_distribution.values())
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=CATEGORICAL_PALETTE[: len(labels)],
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(xaxis_title="Class", yaxis_title="Number of Samples")
    return _base_layout(fig, title)


def plot_missing_values(missing_by_column: dict[str, int], top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of columns with the most missing values."""
    if not missing_by_column:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values detected \u2713",
            showarrow=False,
            font=dict(size=16, color=COLOR_SUCCESS),
        )
        return _base_layout(fig, "Missing Values by Column", height=200)

    series = pd.Series(missing_by_column).sort_values(ascending=True).tail(top_n)
    fig = go.Figure(
        data=[go.Bar(x=series.values, y=series.index, orientation="h", marker_color=COLOR_WARNING)]
    )
    fig.update_layout(xaxis_title="Missing Count", yaxis_title="Column")
    return _base_layout(fig, "Missing Values by Column", height=max(300, 24 * len(series)))


def plot_dtype_breakdown(n_numeric: int, n_non_numeric: int) -> go.Figure:
    """Pie chart of numeric vs non-numeric column counts."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Numeric", "Non-numeric"],
                values=[n_numeric, n_non_numeric],
                hole=0.55,
                marker_colors=[COLOR_PRIMARY, COLOR_MUTED],
            )
        ]
    )
    return _base_layout(fig, "Column Data Types", height=320)


# --------------------------------------------------------------------------
# Feature selection page
# --------------------------------------------------------------------------


def plot_feature_reduction(n_original: int, n_selected: int) -> go.Figure:
    """Bar chart comparing original vs ANOVA-selected feature counts."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Original Features", "Selected Features"],
                y=[n_original, n_selected],
                marker_color=[COLOR_MUTED, COLOR_SUCCESS],
                text=[n_original, n_selected],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(yaxis_title="Number of Features")
    return _base_layout(fig, "Feature Reduction using ANOVA (F-test)")


def plot_feature_scores(feature_scores: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of the top-N ANOVA F-scores."""
    top = feature_scores.head(top_n).sort_values("f_score", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["f_score"],
                y=top["feature"],
                orientation="h",
                marker_color=COLOR_ACCENT,
            )
        ]
    )
    fig.update_layout(xaxis_title="ANOVA F-score", yaxis_title="Gene / Feature")
    return _base_layout(fig, f"Top {top_n} Features by ANOVA F-score", height=max(400, 22 * top_n))


# --------------------------------------------------------------------------
# Classification / Results page
# --------------------------------------------------------------------------


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str = "Confusion Matrix") -> go.Figure:
    """Annotated heatmap confusion matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=14),
            showscale=True,
        )
    )
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed"),
    )
    return _base_layout(fig, title, height=420)


def plot_accuracy_comparison(classical_accuracy: float, quantum_accuracy: float) -> go.Figure:
    """Bar chart comparing classical vs quantum-enhanced SVM accuracy."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Classical SVM", "Quantum-Enhanced SVM"],
                y=[classical_accuracy, quantum_accuracy],
                marker_color=[CLASSICAL_COLOR, QUANTUM_COLOR],
                text=[f"{classical_accuracy:.2%}", f"{quantum_accuracy:.2%}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(yaxis_title="Accuracy", yaxis_range=[0, 1])
    return _base_layout(fig, "Classical vs Quantum-Enhanced Model Accuracy")


def plot_metric_comparison_grouped(comparison: dict) -> go.Figure:
    """Grouped bar chart comparing accuracy/precision/recall/F1 side by side."""
    metrics = comparison["Metric"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Classical SVM", x=metrics, y=comparison["Classical SVM"], marker_color=CLASSICAL_COLOR)
    )
    fig.add_trace(
        go.Bar(
            name="Quantum-Enhanced SVM",
            x=metrics,
            y=comparison["Quantum-Enhanced SVM"],
            marker_color=QUANTUM_COLOR,
        )
    )
    fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1])
    return _base_layout(fig, "Classical vs Quantum: Full Metric Comparison")


def plot_roc_curves(y_test: np.ndarray, y_proba: np.ndarray, target_names: list[str], title: str) -> Optional[go.Figure]:
    """One-vs-rest multiclass ROC curves. Returns None if not computable."""
    if y_proba is None:
        return None
    classes = list(range(len(target_names)))
    y_bin = label_binarize(y_test, classes=classes)
    if y_bin.shape[1] == 1:
        # Binary case: label_binarize collapses to a single column.
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=COLOR_MUTED), name="Chance"))

    for i, class_name in enumerate(target_names):
        if i >= y_proba.shape[1]:
            break
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{class_name} (AUC={roc_auc:.2f})",
                line=dict(width=2, color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]),
            )
        )

    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _base_layout(fig, title)


def plot_classification_report_heatmap(report_dict: dict, title: str) -> go.Figure:
    """Heatmap of per-class precision/recall/F1 from a sklearn classification report."""
    classes = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]
    z = [[report_dict[c][m] for m in metrics] for c in classes]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=metrics,
            y=classes,
            colorscale="Purples",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
        )
    )
    return _base_layout(fig, title, height=max(280, 60 * len(classes)))


# --------------------------------------------------------------------------
# Quantum analysis page
# --------------------------------------------------------------------------


def plot_quantum_histogram(counts: dict[str, int], title: str = "Quantum Measurement Distribution") -> go.Figure:
    """Bar chart of measured bitstring counts (a Plotly analogue of
    Qiskit's plot_histogram, styled to match the rest of the dashboard).
    """
    states = sorted(counts.keys())
    values = [counts[s] for s in states]
    fig = go.Figure(
        data=[go.Bar(x=states, y=values, marker_color=COLOR_SECONDARY)]
    )
    fig.update_layout(xaxis_title="Measured Basis State", yaxis_title="Counts")
    return _base_layout(fig, title)


def plot_quantum_feature_vector(vector: list[float], title: str = "Quantum Feature Vector") -> go.Figure:
    """Bar chart of a single sample's derived quantum feature vector."""
    n = len(vector)
    labels = [format(i, f"0{max(1, n.bit_length() - 1)}b") for i in range(n)]
    fig = go.Figure(data=[go.Bar(x=labels, y=vector, marker_color=COLOR_ACCENT)])
    fig.update_layout(xaxis_title="Basis State Index", yaxis_title="Count")
    return _base_layout(fig, title)


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, target_names: list[str], title: str) -> go.Figure:
    """2D PCA projection scatter plot, used to visually compare classical
    vs quantum feature spaces on the Results page.
    """
    from sklearn.decomposition import PCA

    n_components = min(2, X.shape[1])
    if n_components < 2:
        # Pad with a zero column so a 2D scatter can still render.
        X = np.hstack([X, np.zeros((X.shape[0], 1))])
        n_components = 2

    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    labels = [target_names[i] for i in y]

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=labels,
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"x": "Principal Component 1", "y": "Principal Component 2", "color": "Class"},
    )
    return _base_layout(fig, title)


# --------------------------------------------------------------------------
# Home / architecture overview
# --------------------------------------------------------------------------


def plot_pipeline_funnel(stage_labels: list[str], stage_values: list[int], title: str = "Pipeline Data Flow") -> go.Figure:
    """Funnel chart summarizing how dimensionality shrinks through the pipeline."""
    fig = go.Figure(
        go.Funnel(
            y=stage_labels,
            x=stage_values,
            textinfo="value+percent initial",
            marker=dict(color=[COLOR_PRIMARY, COLOR_ACCENT, COLOR_SECONDARY, COLOR_SUCCESS][: len(stage_labels)]),
        )
    )
    return _base_layout(fig, title, height=380)
