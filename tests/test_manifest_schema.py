from __future__ import annotations

import json
from pathlib import Path

from dmvc_poc.data.schema import validate_clip_manifest


def test_sample_fixture_validates_against_schema(repo_root: Path) -> None:
    path = repo_root / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    instance = json.loads(path.read_text(encoding="utf-8"))
    validate_clip_manifest(instance)
