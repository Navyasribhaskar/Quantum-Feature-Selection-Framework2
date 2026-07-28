"""
utils.py
========
Shared constants, configuration, logging helpers, and small utility
functions used throughout the Quantum Feature Selection Framework.

This module intentionally has no heavy dependencies (no sklearn/qiskit)
so it can be imported cheaply from every page of the dashboard.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for the application."""
    return logging.getLogger(name)


log = get_logger("qfsf")

# --------------------------------------------------------------------------
# Project-wide constants
# --------------------------------------------------------------------------

# Columns in the raw Golub et al. leukemia dataset that carry clinical /
# administrative metadata rather than gene-expression signal. These are
# dropped prior to modelling. The pipeline also tolerates datasets that do
# not contain some (or any) of these columns.
DEFAULT_METADATA_COLUMNS: list[str] = [
    "Samples",
    "BM.PB",
    "Gender",
    "Source",
    "tissue.mf",
]

# Name of the target / label column in the source dataset.
DEFAULT_TARGET_COLUMN: str = "cancer"

# Feature selection
DEFAULT_TOP_K_FEATURES: int = 100

# Quantum encoding
DEFAULT_N_QUBITS: int = 4
DEFAULT_SHOTS: int = 256
DEMO_SHOTS: int = 1024

# Modelling
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_SVM_KERNEL: str = "linear"

# Paths (relative to the dashboard's working directory, which is the
# project root when launched via `streamlit run dashboard/app.py`)
DATA_RAW_DIR = os.path.join("data", "raw")
DATA_SAMPLE_DIR = os.path.join("data", "sample")
OUTPUTS_DIR = "outputs"
ASSETS_DIR = "assets"
DEFAULT_DATASET_PATH = os.path.join(DATA_RAW_DIR, "golub.csv")

APP_TITLE = "Quantum Feature Selection Framework"
APP_SUBTITLE = "Hybrid Quantum-Classical Machine Learning for Gene Expression Analysis"

# --------------------------------------------------------------------------
# Session-state keys (centralised so pages never typo a key name)
# --------------------------------------------------------------------------


class SKeys:
    """Canonical st.session_state key names used across dashboard pages."""

    RAW_DF = "raw_df"
    DATASET_NAME = "dataset_name"
    DATASET_INFO = "dataset_info"

    METADATA_COLUMNS = "metadata_columns"
    TARGET_COLUMN = "target_column"

    X_RAW = "X_raw"
    Y_RAW = "y_raw"

    X_TRAIN = "X_train"
    X_TEST = "X_test"
    Y_TRAIN = "y_train"
    Y_TEST = "y_test"
    FEATURE_NAMES = "feature_names"
    LABEL_ENCODER = "label_encoder"
    SCALER = "scaler"
    PREPROCESS_LOG = "preprocess_log"
    PREPROCESS_DONE = "preprocess_done"

    SELECTOR = "selector"
    X_TRAIN_SELECTED = "X_train_selected"
    X_TEST_SELECTED = "X_test_selected"
    SELECTED_FEATURE_NAMES = "selected_feature_names"
    FEATURE_SCORES = "feature_scores"
    TOP_K = "top_k_features"
    FEATURE_SELECTION_DONE = "feature_selection_done"

    N_QUBITS = "n_qubits"
    SHOTS = "shots"
    X_TRAIN_QUANTUM_RAW = "X_train_quantum_raw"
    X_TEST_QUANTUM_RAW = "X_test_quantum_raw"
    X_TRAIN_QUANTUM_FEATURES = "X_train_quantum_features"
    X_TEST_QUANTUM_FEATURES = "X_test_quantum_features"
    SAMPLE_CIRCUIT_COUNTS = "sample_circuit_counts"
    QUANTUM_ANALYSIS_DONE = "quantum_analysis_done"

    CLASSICAL_MODEL = "classical_model"
    QUANTUM_MODEL = "quantum_model"
    CLASSICAL_METRICS = "classical_metrics"
    QUANTUM_METRICS = "quantum_metrics"
    CLASSIFICATION_DONE = "classification_done"


# --------------------------------------------------------------------------
# Small dataclasses for passing structured results between src/ and pages/
# --------------------------------------------------------------------------


@dataclass
class DatasetInfo:
    """Summary statistics describing an uploaded / loaded dataset."""

    n_rows: int
    n_columns: int
    n_numeric_columns: int
    n_non_numeric_columns: int
    missing_values_total: int
    missing_by_column: dict[str, int]
    memory_usage_mb: float
    class_distribution: dict[str, int] = field(default_factory=dict)
    target_column: Optional[str] = None


@dataclass
class StepLogEntry:
    """A single, human-readable record of a pipeline step (for UI display)."""

    step: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def bytes_to_mb(n_bytes: float) -> float:
    """Convert a byte count to megabytes, rounded to 3 decimals."""
    return round(n_bytes / (1024 ** 2), 3)


def dataframe_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """Serialize a DataFrame to CSV bytes suitable for st.download_button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=index)
    return buffer.getvalue().encode("utf-8")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning `default` instead of raising on /0."""
    if denominator == 0:
        return default
    return numerator / denominator


def percentage_reduction(original: int, reduced: int) -> float:
    """Percentage reduction from `original` count to `reduced` count."""
    if original == 0:
        return 0.0
    return round((1 - (reduced / original)) * 100, 2)


def format_seconds(seconds: float) -> str:
    """Human-friendly duration string, e.g. '1m 03s' or '4.2s'."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def ensure_dir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def is_binary_or_multiclass(y: np.ndarray) -> str:
    """Return 'binary' or 'multiclass' based on the number of unique labels."""
    n_classes = len(np.unique(y))
    return "binary" if n_classes == 2 else "multiclass"


class PipelineStateError(RuntimeError):
    """Raised when a dashboard page is visited out of order.

    For example, opening the Feature Selection page before a dataset has
    been preprocessed. Pages catch this and render a friendly warning that
    redirects the user to the correct earlier step, instead of crashing
    with a raw KeyError/AttributeError.
    """
