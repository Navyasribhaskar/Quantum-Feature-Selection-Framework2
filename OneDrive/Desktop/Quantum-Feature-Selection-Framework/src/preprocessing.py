"""
preprocessing.py
================
Data cleaning, label encoding, feature scaling, and train/test splitting.

This module is a direct, modularized extraction of the preprocessing
cells from the source notebook (`01_Project_Foundation_ipynb.ipynb`):

    1. Drop clinical metadata columns (Samples, BM.PB, Gender, Source,
       tissue.mf) that are not gene-expression features.
    2. Handle missing values.
    3. Separate features (X) from the target label (y).
    4. Label-encode the (categorical) target.
    5. Standardize features with StandardScaler.
    6. Stratified train/test split.

The notebook's own logic and parameters (test_size=0.2, random_state=42,
StandardScaler, LabelEncoder) are preserved exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    StepLogEntry,
    get_logger,
)

log = get_logger(__name__)


def drop_metadata_columns(df: pd.DataFrame, metadata_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop clinical/administrative metadata columns from the dataset.

    Only columns that are actually present are dropped, so this is safe
    to call on datasets that don't contain the full original metadata set.

    Returns
    -------
    (cleaned_df, dropped_columns)
    """
    present = [col for col in metadata_columns if col in df.columns]
    cleaned = df.drop(columns=present)
    log.info("Dropped metadata columns: %s", present)
    return cleaned, present


def handle_missing_values(df: pd.DataFrame, strategy: str = "zero") -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill missing values in a feature DataFrame.

    Parameters
    ----------
    df:
        Feature DataFrame (target column should already be removed).
    strategy:
        'zero'   -> fill with 0 (matches the source notebook's `fillna(0)`)
        'mean'   -> fill numeric columns with their column mean
        'median' -> fill numeric columns with their column median

    Returns
    -------
    (filled_df, missing_counts_before) where missing_counts_before maps
    column name -> number of missing values that were present.
    """
    missing_before = {
        col: int(count) for col, count in df.isnull().sum().items() if count > 0
    }

    if strategy == "zero":
        filled = df.fillna(0)
    elif strategy == "mean":
        filled = df.fillna(df.mean(numeric_only=True))
        filled = filled.fillna(0)  # any remaining non-numeric NaNs
    elif strategy == "median":
        filled = df.fillna(df.median(numeric_only=True))
        filled = filled.fillna(0)
    else:
        raise ValueError(f"Unknown missing-value strategy: '{strategy}'")

    log.info("Filled missing values (strategy=%s). Columns affected: %s", strategy, missing_before)
    return filled, missing_before


def split_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Label-encode a categorical target vector.

    Returns
    -------
    (y_encoded, fitted_label_encoder)
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    log.info("Encoded labels. Class mapping: %s", dict(enumerate(label_encoder.classes_)))
    return y_encoded, label_encoder


def scale_features(
    X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None
) -> tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Standardize features (zero mean, unit variance).

    Fits the scaler on X_train only, to avoid test-set leakage, then
    applies the fitted transform to X_test if provided.

    If X_test is None, both training-only scaling (as done once in the
    notebook before splitting) is still supported by passing the full
    matrix as X_train and leaving X_test as None.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def train_test_split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split, matching the source notebook's call."""
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
    metadata_columns: list[str],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    missing_value_strategy: str = "zero",
) -> dict:
    """Run the full preprocessing pipeline end-to-end and return all artifacts.

    This orchestrates, in order, the same steps performed in the source
    notebook: drop metadata -> handle missing values -> separate X/y ->
    encode labels -> scale features -> stratified train/test split.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test (np.ndarray)
        feature_names (list[str])
        label_encoder (LabelEncoder)
        scaler (StandardScaler)
        dropped_metadata_columns (list[str])
        missing_value_counts (dict[str, int])
        steps (list[StepLogEntry])  -- for UI display
    """
    steps: list[StepLogEntry] = []

    cleaned_df, dropped_cols = drop_metadata_columns(df, metadata_columns)
    steps.append(
        StepLogEntry(
            step="Remove Metadata Columns",
            detail=(
                f"Dropped {len(dropped_cols)} metadata column(s): "
                f"{', '.join(dropped_cols) if dropped_cols else 'none present'}"
            ),
        )
    )

    X_df, y = split_features_and_target(cleaned_df, target_column)
    steps.append(
        StepLogEntry(
            step="Separate Features and Target",
            detail=f"Feature matrix shape {X_df.shape}, target shape {y.shape}",
        )
    )

    X_df_filled, missing_counts = handle_missing_values(X_df, strategy=missing_value_strategy)
    steps.append(
        StepLogEntry(
            step="Handle Missing Values",
            detail=(
                f"Filled {sum(missing_counts.values())} missing value(s) across "
                f"{len(missing_counts)} column(s) using '{missing_value_strategy}' strategy"
                if missing_counts
                else "No missing values found in the feature matrix"
            ),
        )
    )

    y_encoded, label_encoder = encode_labels(y)
    class_mapping = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    steps.append(
        StepLogEntry(
            step="Label Encoding",
            detail=f"Class mapping: {class_mapping}",
        )
    )

    X_scaled, _, scaler = scale_features(X_df_filled)
    steps.append(
        StepLogEntry(
            step="Feature Standardization",
            detail=f"Standardized {X_scaled.shape[1]} features to zero mean / unit variance",
        )
    )

    X_train, X_test, y_train, y_test = train_test_split_data(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    steps.append(
        StepLogEntry(
            step="Train/Test Split",
            detail=(
                f"Training shape {X_train.shape}, testing shape {X_test.shape} "
                f"(test_size={test_size}, stratified, random_state={random_state})"
            ),
        )
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X_df.columns),
        "label_encoder": label_encoder,
        "scaler": scaler,
        "dropped_metadata_columns": dropped_cols,
        "missing_value_counts": missing_counts,
        "class_mapping": class_mapping,
        "steps": steps,
    }
