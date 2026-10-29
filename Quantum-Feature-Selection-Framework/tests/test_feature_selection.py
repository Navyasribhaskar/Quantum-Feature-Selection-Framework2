"""Unit tests for src/feature_selection.py."""

import numpy as np

from src.feature_selection import select_features_anova


def _make_synthetic_data(n_samples=40, n_features=20, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # first two features are informative
    feature_names = [f"gene_{i}" for i in range(n_features)]
    split = n_samples // 2
    return X[:split], X[split:], y[:split], y[split:], feature_names


def test_select_features_anova_shapes():
    X_train, X_test, y_train, y_test, feature_names = _make_synthetic_data()
    result = select_features_anova(X_train, X_test, y_train, feature_names, k=5)

    assert result["X_train_selected"].shape == (X_train.shape[0], 5)
    assert result["X_test_selected"].shape == (X_test.shape[0], 5)
    assert len(result["selected_feature_names"]) == 5
    assert result["n_original_features"] == 20
    assert result["n_selected_features"] == 5


def test_select_features_anova_k_larger_than_features_is_capped():
    X_train, X_test, y_train, y_test, feature_names = _make_synthetic_data(n_features=5)
    result = select_features_anova(X_train, X_test, y_train, feature_names, k=100)

    assert result["n_selected_features"] == 5
    assert result["X_train_selected"].shape[1] == 5


def test_feature_scores_dataframe_sorted_descending():
    X_train, X_test, y_train, y_test, feature_names = _make_synthetic_data()
    result = select_features_anova(X_train, X_test, y_train, feature_names, k=5)

    scores = result["feature_scores"]["f_score"].to_numpy()
    assert (scores[:-1] >= scores[1:]).all()
