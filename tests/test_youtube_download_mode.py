from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_full_download_optin_simulation(repo_root: Path, tmp_path: Path) -> None:
    script = repo_root / "scripts" / "youtube_metadata_crawl.py"
    mock_rows = repo_root / "fixtures" / "synthetic" / "candidate_rows_mock.json"
    out_dir = tmp_path / "candidates"
    download_dir = tmp_path / "raw_video"
    cmd = [
        sys.executable,
        str(script),
        "--mock-json",
        str(mock_rows),
        "--out-dir",
        str(out_dir),
        "--download-dir",
        str(download_dir),
        "--enable-full-download",
        "--allow-risky-download",
        "--simulate-download",
    ]
    subprocess.run(cmd, check=True)
    consent = out_dir / "full_download_consent.log"
    assert consent.is_file()
    json_files = sorted(out_dir.glob("candidates_*.json"))
    assert json_files
    payload = json.loads(json_files[-1].read_text(encoding="utf-8"))
    assert payload[0]["download_status"] == "downloaded"
    assert payload[0]["local_video_path"]
    assert Path(payload[0]["local_video_path"]).is_file()
