"""
feature_selection.py
=====================
Classical feature selection using ANOVA F-test (SelectKBest + f_classif),
exactly as implemented in the source notebook.

Gene-expression datasets are extremely wide (thousands of genes for tens
of samples), so this step reduces dimensionality to the top-K most
statistically significant features before quantum encoding and
classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from src.utils import DEFAULT_TOP_K_FEATURES, get_logger, percentage_reduction

log = get_logger(__name__)


def select_features_anova(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    k: int = DEFAULT_TOP_K_FEATURES,
) -> dict:
    """Select the top-`k` features using an ANOVA F-test.

    Mirrors the notebook's:
        selector = SelectKBest(score_func=f_classif, k=100)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    Returns a dict with:
        selector                 (fitted SelectKBest)
        X_train_selected          (np.ndarray)
        X_test_selected           (np.ndarray)
        selected_feature_names    (list[str])
        feature_scores            (pd.DataFrame: feature, f_score, p_value, selected)
        n_original_features       (int)
        n_selected_features       (int)
        reduction_percentage      (float)
    """
    k = min(k, X_train.shape[1])  # guard against k > available features

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    support_mask = selector.get_support()
    selected_feature_names = [
        name for name, keep in zip(feature_names, support_mask) if keep
    ]

    scores = np.nan_to_num(selector.scores_, nan=0.0)
    pvalues = np.nan_to_num(selector.pvalues_, nan=1.0)

    feature_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "f_score": scores,
            "p_value": pvalues,
            "selected": support_mask,
        }
    ).sort_values("f_score", ascending=False).reset_index(drop=True)

    n_original = len(feature_names)
    n_selected = len(selected_feature_names)

    log.info(
        "ANOVA feature selection: %d -> %d features (%.2f%% reduction)",
        n_original,
        n_selected,
        percentage_reduction(n_original, n_selected),
    )

    return {
        "selector": selector,
        "X_train_selected": X_train_selected,
        "X_test_selected": X_test_selected,
        "selected_feature_names": selected_feature_names,
        "feature_scores": feature_scores,
        "n_original_features": n_original,
        "n_selected_features": n_selected,
        "reduction_percentage": percentage_reduction(n_original, n_selected),
    }
