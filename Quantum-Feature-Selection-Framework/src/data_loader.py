"""
data_loader.py
===============
Dataset ingestion, validation, and summary-statistics utilities.

Responsible for turning a raw CSV (uploaded by the user, or the bundled
Golub et al. leukemia gene-expression dataset) into a validated pandas
DataFrame, plus a `DatasetInfo` summary used to populate the Dataset page.
"""

from __future__ import annotations

import os
from typing import IO, Optional, Union

import numpy as np
import pandas as pd

from src.utils import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TARGET_COLUMN,
    DatasetInfo,
    bytes_to_mb,
    get_logger,
)

log = get_logger(__name__)


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset fails structural validation."""


def load_default_dataset(path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the bundled Golub et al. leukemia gene-expression dataset.

    Parameters
    ----------
    path:
        Path to the CSV file, relative to the app's working directory.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Default dataset not found at '{path}'. Make sure the "
            f"'data/raw/golub.csv' file ships with the application."
        )
    df = pd.read_csv(path)
    log.info("Loaded default dataset from %s with shape %s", path, df.shape)
    return df


def load_uploaded_dataset(file: Union[str, IO[bytes]]) -> pd.DataFrame:
    """Load a user-uploaded dataset (Streamlit UploadedFile or path).

    Raises
    ------
    DatasetValidationError
        If the file cannot be parsed as CSV or is empty.
    """
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            "The uploaded file could not be decoded as text. Please upload "
            "a UTF-8 (or Latin-1) encoded CSV file."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError(
            "The uploaded file is empty. Please upload a non-empty CSV file."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(
            f"The uploaded file could not be parsed as CSV: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any other IO error clearly
        raise DatasetValidationError(f"Could not read the uploaded file: {exc}") from exc

    if df.shape[0] == 0:
        raise DatasetValidationError("The uploaded dataset contains zero rows.")
    if df.shape[1] < 2:
        raise DatasetValidationError(
            "The uploaded dataset must contain at least one feature column "
            "and one target/label column."
        )
    log.info("Loaded uploaded dataset with shape %s", df.shape)
    return df


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Ensure the requested target column exists and is usable as a label.

    Raises
    ------
    DatasetValidationError
    """
    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' was not found in the dataset. "
            f"Available columns include: {', '.join(map(str, df.columns[:10]))}..."
        )
    n_unique = df[target_column].nunique(dropna=True)
    if n_unique < 2:
        raise DatasetValidationError(
            f"Target column '{target_column}' has only {n_unique} unique "
            f"class(es). At least 2 classes are required for classification."
        )
    if df[target_column].isnull().all():
        raise DatasetValidationError(
            f"Target column '{target_column}' contains only missing values."
        )


def get_dataset_info(df: pd.DataFrame, target_column: Optional[str] = None) -> DatasetInfo:
    """Compute summary statistics for display on the Dataset page."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.columns.difference(numeric_cols)

    missing_by_column = {
        col: int(count)
        for col, count in df.isnull().sum().items()
        if count > 0
    }

    class_distribution: dict[str, int] = {}
    resolved_target = target_column if target_column in df.columns else None
    if resolved_target is not None:
        class_distribution = {
            str(k): int(v) for k, v in df[resolved_target].value_counts().items()
        }

    return DatasetInfo(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        n_numeric_columns=len(numeric_cols),
        n_non_numeric_columns=len(non_numeric_cols),
        missing_values_total=int(df.isnull().sum().sum()),
        missing_by_column=missing_by_column,
        memory_usage_mb=bytes_to_mb(df.memory_usage(deep=True).sum()),
        class_distribution=class_distribution,
        target_column=resolved_target,
    )


def infer_metadata_columns(df: pd.DataFrame, candidate_columns: list[str]) -> list[str]:
    """Return only the candidate metadata columns that actually exist in df.

    This lets the pipeline gracefully handle datasets that don't include
    every clinical metadata field the original Golub dataset had.
    """
    return [col for col in candidate_columns if col in df.columns]


def suggest_target_column(df: pd.DataFrame) -> str:
    """Best-effort guess at which column is the classification target.

    Prefers a column literally named 'cancer' (matching the source
    notebook); otherwise falls back to the lowest-cardinality
    non-numeric/categorical column, which is a reasonable heuristic for
    a class-label column in a wide gene-expression table.
    """
    if DEFAULT_TARGET_COLUMN in df.columns:
        return DEFAULT_TARGET_COLUMN

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        cardinalities = {col: df[col].nunique(dropna=True) for col in categorical_cols}
        # A good label column has more than 1 class but far fewer unique
        # values than there are rows.
        plausible = {c: n for c, n in cardinalities.items() if 2 <= n <= max(20, df.shape[0] // 2)}
        if plausible:
            return min(plausible, key=plausible.get)
        return min(cardinalities, key=cardinalities.get)

    # Fall back to the last column, a common convention for label columns.
    return df.columns[-1]
