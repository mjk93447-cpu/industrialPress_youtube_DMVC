from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dmvc_poc.data.dataset import ClipManifestIndex, ProcessedClipDataset


def test_preprocess_with_synthetic_fallback(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "scripts" / "preprocess.py"
    manifest_path = repo_root / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    out_dir = tmp_path / "processed"
    cmd = [
        sys.executable,
        str(script),
        "--manifest-json",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
        "--allow-synthetic-fallback",
    ]
    subprocess.run(cmd, check=True)
    log = json.loads((out_dir / "preprocess_log.json").read_text(encoding="utf-8"))
    assert log["processed"] == 1
    assert (out_dir / "synthetic_fixture_001.npz").is_file()


def test_processed_clip_dataset_reads_npz(repo_root: Path, tmp_path: Path) -> None:
    manifest = ClipManifestIndex.from_json_file(repo_root / "fixtures" / "synthetic" / "sample_clip_manifest.json")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_dir / "synthetic_fixture_001.npz",
        video=np.zeros((8, 3, 16, 16), dtype=np.float32),
        audio=np.zeros((1600,), dtype=np.float32),
        frame_diff=np.zeros((8, 3, 16, 16), dtype=np.float32),
    )
    ds = ProcessedClipDataset(list(manifest), processed_dir)
    row = ds[0]
    assert row["video"].shape[0] == 8
