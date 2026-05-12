from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_mvp_smoke_pipeline_end_to_end(repo_root: Path, tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_mvp.py"),
        "--root",
        str(tmp_path),
        "--smoke",
    ]
    # copy fixture dependency into tmp root for smoke mode
    fixture_src = repo_root / "fixtures" / "synthetic" / "candidate_rows_mock.json"
    fixture_dst = tmp_path / "fixtures" / "synthetic" / "candidate_rows_mock.json"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    fixture_dst.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

    manifest_src = repo_root / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    manifest_dst = tmp_path / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    manifest_dst.write_text(manifest_src.read_text(encoding="utf-8"), encoding="utf-8")

    # minimal scripts and package path still come from repo_root
    subprocess.run(cmd, check=True, cwd=repo_root)
    report_path = tmp_path / "artifacts" / "final_mvp_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["gate_status"] in {"PASSED_GATE", "FAILED_GATE"}
    assert Path(payload["metrics_path"]).is_file()
