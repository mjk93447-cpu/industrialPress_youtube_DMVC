#!/usr/bin/env python3
"""One-command local MVP pipeline runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run_step(name: str, cmd: list[str], log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / f"{name}.log"
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Step {name} failed with code {proc.returncode}; see {log_path}")
    return {"name": name, "returncode": proc.returncode, "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full local MVP pipeline")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-full-download", action="store_true")
    parser.add_argument("--allow-risky-download", action="store_true")
    parser.add_argument("--simulate-download", action="store_true")
    parser.add_argument("--queries", nargs="*", default=["hydraulic press crush"])
    args = parser.parse_args()

    root = args.root.resolve()
    code_root = Path(__file__).resolve().parents[1]
    logs_dir = root / "artifacts" / "mvp_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "artifacts" / "mvp_runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates_dir = root / "data" / "candidates"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    demo_dir = root / "artifacts" / "demo"

    py = sys.executable
    steps: list[dict[str, Any]] = []

    crawl_cmd = [py, str(code_root / "scripts" / "youtube_metadata_crawl.py"), "--out-dir", str(candidates_dir)]
    if args.smoke:
        crawl_cmd.extend(
            [
                "--mock-json",
                str(root / "fixtures" / "synthetic" / "candidate_rows_mock.json"),
            ]
        )
    else:
        crawl_cmd.extend(["--queries", *args.queries])
    if args.enable_full_download:
        crawl_cmd.append("--enable-full-download")
        if args.allow_risky_download:
            crawl_cmd.append("--allow-risky-download")
        if args.simulate_download:
            crawl_cmd.append("--simulate-download")
    steps.append(_run_step("01_crawl", crawl_cmd, logs_dir))

    manifest_path = root / "fixtures" / "synthetic" / "sample_clip_manifest.json"
    preprocess_cmd = [
        py,
        str(code_root / "scripts" / "preprocess.py"),
        "--manifest-json",
        str(manifest_path),
        "--out-dir",
        str(processed_dir),
        "--allow-synthetic-fallback",
    ]
    steps.append(_run_step("02_preprocess", preprocess_cmd, logs_dir))

    train_cmd = [
        py,
        str(code_root / "scripts" / "train.py"),
        "--processed-dir",
        str(processed_dir),
        "--output-dir",
        str(root / "artifacts" / "checkpoints"),
        "--epochs",
        "1" if args.smoke else "3",
    ]
    steps.append(_run_step("03_train", train_cmd, logs_dir))

    eval_cmd = [
        py,
        str(code_root / "scripts" / "evaluate.py"),
        "--processed-dir",
        str(processed_dir),
        "--out-dir",
        str(reports_dir),
    ]
    steps.append(_run_step("04_evaluate", eval_cmd, logs_dir))

    demo_cmd = [
        py,
        str(code_root / "scripts" / "demo_infer.py"),
        "--clip",
        "synthetic_fixture_001",
        "--processed-dir",
        str(processed_dir),
        "--out-dir",
        str(demo_dir),
    ]
    steps.append(_run_step("05_demo", demo_cmd, logs_dir))

    metrics_path = reports_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = {
        "run_id": ts,
        "steps": steps,
        "gate_status": metrics.get("acceptance_gate", {}).get("status", "UNKNOWN"),
        "metrics_path": str(metrics_path),
        "demo_outputs": {
            "heatmap_png": str(demo_dir / "synthetic_fixture_001_heatmap.png"),
            "timeline_json": str(demo_dir / "synthetic_fixture_001_timeline.json"),
            "card_png": str(demo_dir / "synthetic_fixture_001_card.png"),
        },
    }
    final_path = root / "artifacts" / "final_mvp_report.json"
    final_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    (run_dir / "final_mvp_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps({"status": final["gate_status"], "final_report": str(final_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
