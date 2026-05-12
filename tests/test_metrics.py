from __future__ import annotations

import math

import numpy as np
import pytest

from dmvc_poc.eval.metrics import binary_auroc, soft_iou


def test_soft_iou_perfect_overlap() -> None:
    x = np.ones((4, 4), dtype=np.float64)
    assert soft_iou(x, x) == pytest.approx(1.0)


def test_binary_auroc_requires_two_classes() -> None:
    y_true = np.zeros(10, dtype=np.int64)
    y_score = np.linspace(0.1, 0.9, 10)
    assert math.isnan(binary_auroc(y_true, y_score))


def test_binary_auroc_sklearn_available() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.2, 0.7, 0.9], dtype=np.float64)
    score = binary_auroc(y_true, y_score)
    assert score == pytest.approx(1.0)
