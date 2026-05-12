#!/usr/bin/env python3
"""Train baseline ladder or DMVC-lite on preprocessed arrays."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dmvc_poc.train import run_training


DEFAULT_MODES = [
    "rgb_temporal_baseline",
    "rgb_motion_baseline",
    "rgb_audio_baseline",
    "tri_modal_baseline",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DMVC baseline ladder")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--mode", choices=DEFAULT_MODES + ["dmvc_lite"], default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    modes = [args.mode] if args.mode else DEFAULT_MODES
    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in modes:
        result = run_training(
            mode=mode,
            processed_dir=args.processed_dir,
            output_dir=run_dir,
            epochs=args.epochs,
            seed=args.seed,
        )
        rows.append(
            {
                "mode": result.mode,
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
                "checkpoint_path": str(result.checkpoint_path),
                "log_path": str(result.log_path),
            }
        )
        print(f"[ok] {mode}: train_loss={result.train_loss:.4f} val_loss={result.val_loss:.4f}")
    (run_dir / "summary.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    print(f"summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
