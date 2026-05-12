"""Visualization helper functions for inference output cards."""

from __future__ import annotations

import numpy as np


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float64)
    x = x - np.min(x)
    denom = np.max(x) + 1e-6
    x = x / denom
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
