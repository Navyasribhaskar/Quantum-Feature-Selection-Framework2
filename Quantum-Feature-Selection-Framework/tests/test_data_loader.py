"""Unit tests for src/data_loader.py — validation logic using in-memory data."""

import io

import pandas as pd
import pytest

from src.data_loader import (
    DatasetValidationError,
    get_dataset_info,
    load_uploaded_dataset,
    validate_target_column,
)


def _csv_bytes(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


def test_load_uploaded_dataset_valid_csv():
    csv = "gene_1,gene_2,label\n1.0,2.0,A\n3.0,4.0,B\n"
    df = load_uploaded_dataset(_csv_bytes(csv))
    assert df.shape == (2, 3)


def test_load_uploaded_dataset_empty_file_raises():
    with pytest.raises(DatasetValidationError):
        load_uploaded_dataset(_csv_bytes(""))


def test_load_uploaded_dataset_single_column_raises():
    with pytest.raises(DatasetValidationError):
        load_uploaded_dataset(_csv_bytes("only_col\n1\n2\n"))


def test_validate_target_column_missing_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(DatasetValidationError):
        validate_target_column(df, "missing_col")


def test_validate_target_column_single_class_raises():
    df = pd.DataFrame({"a": [1, 2], "label": ["A", "A"]})
    with pytest.raises(DatasetValidationError):
        validate_target_column(df, "label")


def test_validate_target_column_valid_passes():
    df = pd.DataFrame({"a": [1, 2], "label": ["A", "B"]})
    validate_target_column(df, "label")  # should not raise


def test_get_dataset_info_counts():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    info = get_dataset_info(df, target_column="b")
    assert info.n_rows == 3
    assert info.n_columns == 2
    assert info.n_numeric_columns == 1
    assert info.missing_values_total == 1
