from __future__ import annotations

from pathlib import Path

from dmvc_poc.data.dataset import ClipManifestIndex


def test_clip_manifest_index_loads_fixture(repo_root: Path) -> None:
    path = repo_root / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    index = ClipManifestIndex.from_json_file(path)
    assert len(index) == 1
    clip = index[0]
    assert clip["clip_id"] == "synthetic_fixture_001"
