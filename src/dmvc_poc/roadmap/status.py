"""Roadmap checklist used by tests and scripts (docs/05, docs/07)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoadmapItem:
    """Single verifiable deliverable."""

    phase: str
    item_id: str
    description: str
    paths: tuple[str, ...]
    optional: bool = False


def roadmap_items(repo_root: Path) -> list[RoadmapItem]:
    """Machine-readable checklist; paths are relative to repository root."""
    _ = repo_root  # reserved for future conditional items
    return [
        RoadmapItem(
            "phase0",
            "packaging",
            "Python packaging (pyproject/setuptools src layout)",
            ("pyproject.toml", "src/dmvc_poc/__init__.py"),
        ),
        RoadmapItem(
            "phase0",
            "schema_validation",
            "Clip manifest validation against JSON Schema",
            ("src/dmvc_poc/data/schema.py", "schemas/dmvc_clip.schema.json"),
        ),
        RoadmapItem(
            "phase0",
            "smoke_train",
            "CPU smoke training loop on synthetic tensors",
            ("src/dmvc_poc/train/smoke.py", "src/dmvc_poc/models/rgb_baseline.py"),
        ),
        RoadmapItem(
            "phase0",
            "synthetic_fixture",
            "Committed synthetic manifest fixture",
            ("fixtures/synthetic/sample_clip_manifest.json",),
        ),
        RoadmapItem(
            "phase0",
            "pytest",
            "Automated tests",
            (
                "tests/test_manifest_schema.py",
                "tests/test_smoke_training.py",
                "tests/test_roadmap_compliance.py",
            ),
        ),
        RoadmapItem(
            "phase1",
            "dataset_loader",
            "Manifest-backed dataset index (decode frames/audio in later milestone)",
            ("src/dmvc_poc/data/dataset.py",),
            optional=False,
        ),
        RoadmapItem(
            "phase1",
            "preprocess",
            "Frame/audio preprocessing CLI (planned)",
            ("scripts/preprocess.py",),
            optional=True,
        ),
        RoadmapItem(
            "phase2",
            "train_cli",
            "Training entrypoint script (planned)",
            ("scripts/train.py",),
            optional=True,
        ),
        RoadmapItem(
            "phase3",
            "evaluate_cli",
            "Evaluation script exporting metrics (planned)",
            ("scripts/evaluate.py",),
            optional=True,
        ),
        RoadmapItem(
            "phase4",
            "infer_cli",
            "Inference demo CLI (planned)",
            ("scripts/demo_infer.py",),
            optional=True,
        ),
    ]


def compute_completion(repo_root: Path) -> tuple[float, list[tuple[RoadmapItem, bool]]]:
    """Return fraction of required items whose paths exist."""
    results: list[tuple[RoadmapItem, bool]] = []
    required_hits = 0
    required_total = 0
    for item in roadmap_items(repo_root):
        exists = all((repo_root / p).is_file() for p in item.paths)
        results.append((item, exists))
        if not item.optional:
            required_total += 1
            if exists:
                required_hits += 1
    ratio = required_hits / required_total if required_total else 0.0
    return ratio, results
