from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_youtube_metadata_crawl_with_mock_rows(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "scripts" / "youtube_metadata_crawl.py"
    mock_rows = repo_root / "fixtures" / "synthetic" / "candidate_rows_mock.json"
    out_dir = tmp_path / "candidates"
    cmd = [
        sys.executable,
        str(script),
        "--mock-json",
        str(mock_rows),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    json_files = sorted(out_dir.glob("candidates_*.json"))
    assert json_files, "crawler did not generate JSON output"
    payload = json.loads(json_files[-1].read_text(encoding="utf-8"))
    # duplicate video_id should be deduped
    assert len(payload) == 1
    assert payload[0]["video_id"] == "abc123"
