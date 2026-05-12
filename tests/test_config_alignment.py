from __future__ import annotations

import json
from pathlib import Path


def test_dmvc_poc_model_config_matches_expected_contract(repo_root: Path) -> None:
    """Smoke-check that configs/dmvc_poc_model.json keeps baseline ladder keys."""
    path = repo_root / "configs" / "dmvc_poc_model.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert "baseline_ladder" in cfg
    ladder = cfg["baseline_ladder"]
    assert "rgb_temporal_baseline" in ladder
    assert cfg["views"]["rgb"]["enabled"] is True


def test_evaluation_metrics_config_has_strict_controls(repo_root: Path) -> None:
    path = repo_root / "configs" / "evaluation_metrics.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    strict = cfg["strict_test_controls"]
    assert strict["post_failure_frames_allowed"] is False
