from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_processed_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        video=np.random.randn(12, 3, 16, 16).astype(np.float32),
        audio=np.random.randn(8000).astype(np.float32),
        frame_diff=np.random.randn(12, 3, 16, 16).astype(np.float32),
    )


def test_train_evaluate_demo_pipeline(repo_root: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _make_processed_clip(processed / "synthetic_fixture_001.npz")
    _make_processed_clip(processed / "synthetic_fixture_002.npz")

    train_script = repo_root / "scripts" / "train.py"
    ckpt_out = tmp_path / "checkpoints"
    subprocess.run(
        [
            sys.executable,
            str(train_script),
            "--processed-dir",
            str(processed),
            "--output-dir",
            str(ckpt_out),
            "--mode",
            "rgb_temporal_baseline",
            "--epochs",
            "1",
        ],
        check=True,
    )
    summary_files = list(ckpt_out.rglob("summary.json"))
    assert summary_files

    eval_script = repo_root / "scripts" / "evaluate.py"
    reports = tmp_path / "reports"
    subprocess.run(
        [
            sys.executable,
            str(eval_script),
            "--processed-dir",
            str(processed),
            "--out-dir",
            str(reports),
            "--modes",
            "rgb_temporal_baseline",
            "dmvc_lite",
        ],
        check=True,
    )
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_samples"] == 2
    assert "dmvc_lite" in metrics["modes"]

    demo_script = repo_root / "scripts" / "demo_infer.py"
    demo_out = tmp_path / "demo"
    subprocess.run(
        [
            sys.executable,
            str(demo_script),
            "--clip",
            "synthetic_fixture_001",
            "--processed-dir",
            str(processed),
            "--out-dir",
            str(demo_out),
        ],
        check=True,
    )
    assert (demo_out / "synthetic_fixture_001_heatmap.png").is_file()
    assert (demo_out / "synthetic_fixture_001_timeline.json").is_file()
