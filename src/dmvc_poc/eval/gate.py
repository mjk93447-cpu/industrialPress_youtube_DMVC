"""Acceptance gate checks for MVP metric thresholds."""

from __future__ import annotations

import math
from typing import Any


def compute_gate_status(
    *,
    metrics_by_mode: dict[str, dict[str, Any]],
    mae_max: float,
    auroc_min: float,
    f1_min: float,
) -> dict[str, bool | str]:
    def _is_valid(x: float) -> bool:
        return not math.isnan(x)

    mae_ok = all(
        _is_valid(float(v["mae_seconds"])) and float(v["mae_seconds"]) <= mae_max
        for v in metrics_by_mode.values()
    )
    auroc_ok = any(
        _is_valid(float(v["auroc"])) and float(v["auroc"]) >= auroc_min
        for v in metrics_by_mode.values()
    )
    f1_ok = any(
        _is_valid(float(v["f1"])) and float(v["f1"]) >= f1_min
        for v in metrics_by_mode.values()
    )
    passed = mae_ok and auroc_ok and f1_ok
    return {
        "mae_seconds_le_target_all_modes": mae_ok,
        "validation_auroc_ge_target_any_mode": auroc_ok,
        "test_f1_ge_target_any_mode": f1_ok,
        "passed": passed,
        "status": "PASSED_GATE" if passed else "FAILED_GATE",
    }
