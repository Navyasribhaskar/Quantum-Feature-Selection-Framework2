"""Unit tests for src/utils.py — pure helper functions with no I/O."""

import numpy as np

from src.utils import (
    bytes_to_mb,
    format_seconds,
    is_binary_or_multiclass,
    percentage_reduction,
    safe_divide,
)


def test_bytes_to_mb():
    assert bytes_to_mb(1024 ** 2) == 1.0
    assert bytes_to_mb(0) == 0.0


def test_safe_divide_normal():
    assert safe_divide(10, 2) == 5.0


def test_safe_divide_by_zero_returns_default():
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 0, default=-1) == -1


def test_percentage_reduction():
    assert percentage_reduction(1000, 100) == 90.0
    assert percentage_reduction(0, 0) == 0.0


def test_format_seconds_under_a_minute():
    assert format_seconds(4.2) == "4.20s"


def test_format_seconds_over_a_minute():
    assert format_seconds(63) == "1m 03s"


def test_is_binary_or_multiclass():
    assert is_binary_or_multiclass(np.array([0, 1, 0, 1])) == "binary"
    assert is_binary_or_multiclass(np.array([0, 1, 2, 1])) == "multiclass"
