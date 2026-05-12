from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_demo_infer_writes_standardized_outputs(repo_root: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "sample_clip.npz",
        video=np.random.randn(12, 3, 16, 16).astype(np.float32),
        frame_diff=np.random.randn(12, 3, 16, 16).astype(np.float32),
        audio=np.random.randn(8000).astype(np.float32),
    )
    out_dir = tmp_path / "demo"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "demo_infer.py"),
        "--clip",
        "sample_clip",
        "--processed-dir",
        str(processed),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "sample_clip_heatmap.png").is_file()
    assert (out_dir / "sample_clip_timeline.json").is_file()
    assert (out_dir / "sample_clip_card.png").is_file()
    assert (out_dir / "sample_clip_latent.npy").is_file()
    timeline = json.loads((out_dir / "sample_clip_timeline.json").read_text(encoding="utf-8"))
    assert "pre_failure_probability" in timeline
