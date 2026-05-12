from __future__ import annotations

from dmvc_poc.eval import compute_gate_status


def test_gate_passes_when_all_thresholds_met() -> None:
    metrics = {
        "dmvc_lite": {"mae_seconds": 0.8, "auroc": 0.84, "f1": 0.76},
        "rgb_temporal_baseline": {"mae_seconds": 0.95, "auroc": 0.81, "f1": 0.75},
    }
    gate = compute_gate_status(metrics_by_mode=metrics, mae_max=1.0, auroc_min=0.8, f1_min=0.75)
    assert gate["passed"] is True
    assert gate["status"] == "PASSED_GATE"


def test_gate_fails_when_thresholds_missed() -> None:
    metrics = {
        "dmvc_lite": {"mae_seconds": 1.2, "auroc": 0.7, "f1": 0.6},
    }
    gate = compute_gate_status(metrics_by_mode=metrics, mae_max=1.0, auroc_min=0.8, f1_min=0.75)
    assert gate["passed"] is False
    assert gate["status"] == "FAILED_GATE"
