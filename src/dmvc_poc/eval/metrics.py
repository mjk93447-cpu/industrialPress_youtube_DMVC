"""Evaluation helpers aligned with docs/04_evaluation_plan.md (subset for PoC code)."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
except ImportError:  # pragma: no cover - optional at runtime
    roc_auc_score = None
    average_precision_score = None
    f1_score = None


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


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    num = 2.0 * float((pred * target).sum())
    den = float(pred.sum() + target.sum())
    if den <= eps:
        return 1.0
    return float(num / (den + eps))


def binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if average_precision_score is None:
        raise RuntimeError("scikit-learn is required for binary_auprc")
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    if f1_score is None:
        raise RuntimeError("scikit-learn is required for binary_f1")
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = (np.asarray(y_score).astype(np.float64) >= threshold).astype(np.int64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(f1_score(y_true, y_pred))


def mae_seconds(y_true_seconds: np.ndarray, y_pred_seconds: np.ndarray) -> float:
    y_true = np.asarray(y_true_seconds, dtype=np.float64)
    y_pred = np.asarray(y_pred_seconds, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def median_absolute_error_seconds(y_true_seconds: np.ndarray, y_pred_seconds: np.ndarray) -> float:
    y_true = np.asarray(y_true_seconds, dtype=np.float64)
    y_pred = np.asarray(y_pred_seconds, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.median(np.abs(y_true - y_pred)))


def bootstrap_ci(values: np.ndarray, repeats: int = 200, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(np.mean(sample))
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)
