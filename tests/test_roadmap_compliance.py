from __future__ import annotations

from pathlib import Path

from dmvc_poc.roadmap.status import compute_completion, roadmap_items


def test_phase_zero_required_paths_exist(repo_root: Path) -> None:
    ratio, results = compute_completion(repo_root)
    phase_zero = [(r, ok) for r, ok in results if r.phase == "phase0"]
    assert all(ok for _, ok in phase_zero), [
        (r.item_id, r.paths) for r, ok in phase_zero if not ok
    ]
    assert ratio >= 1.0


def test_future_phases_have_optional_placeholders(repo_root: Path) -> None:
    optional_ids = {i.item_id for i in roadmap_items(repo_root) if i.optional}
    assert {"train_cli", "evaluate_cli", "infer_cli"}.issubset(optional_ids)
