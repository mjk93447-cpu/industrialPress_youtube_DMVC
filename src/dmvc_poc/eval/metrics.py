"""Evaluation helpers aligned with docs/04_evaluation_plan.md (subset for PoC code)."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - optional at runtime
    roc_auc_score = None


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under ROC for binary labels; returns nan if undefined."""
    if roc_auc_score is None:
        raise RuntimeError("scikit-learn is required for binary_auroc")
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def soft_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Soft IoU for probabilistic maps in ``[0, 1]``."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    inter = float((pred * target).sum())
    union = float(pred.sum() + target.sum() - inter)
    if union <= eps and inter <= eps:
        return 1.0
    return float(inter / (union + eps))
