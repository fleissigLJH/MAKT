"""
Evaluation metrics for the MAKT model.

All functions take flattened NumPy arrays of ground-truth labels and
predicted probabilities (plus an optional valid-position mask) and return
Python floats.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


def compute_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    """Binary classification accuracy (default threshold = 0.5)."""
    y_pred = (y_score >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def evaluate_all(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics at once (AUC / ACC / F1 / RMSE / MAE).

    Args:
        y_true  : flattened ground-truth labels
        y_score : flattened predicted probabilities
        mask    : optional flattened mask of valid positions

    Returns:
        dict with keys "auc", "acc", "f1", "rmse", "mae"
    """
    if mask is not None:
        valid = mask > 0
        y_true = y_true[valid]
        y_score = y_score[valid]

    if y_true.size == 0:
        return {"auc": 0.0, "acc": 0.0, "f1": 0.0, "rmse": 0.0, "mae": 0.0}

    y_pred = (y_score >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        # Single-class batch (e.g. a very small split)
        auc = 0.0

    return {
        "auc": auc,
        "acc": float(np.mean(y_pred == y_true)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_score))),
        "mae": float(mean_absolute_error(y_true, y_score)),
    }
