#!/usr/bin/env python3
"""Preprocess approved clips into trainable arrays with robust fallback logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dmvc_poc.data.dataset import ClipManifestIndex


def _load_manifest_list(path: Path) -> list[dict[str, Any]]:
    index = ClipManifestIndex.from_json_file(path)
    return list(index)


def _synthetic_video(seed: int, t: int = 24, h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(t, 3, h, w)).astype(np.float32)


def _synthetic_audio(seed: int, n: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, size=(n,)).astype(np.float32)


def _load_or_fallback(path: Path, fallback: np.ndarray | None) -> np.ndarray | None:
    if path.is_file():
        return np.load(path)
    return fallback


def _crop_before_break(video: np.ndarray, manifest: dict[str, Any], max_frames: int) -> np.ndarray:
    t_break = manifest.get("event_labels", {}).get("t_break", {}).get("frame_index")
    if isinstance(t_break, int) and t_break > 1:
        clipped = video[: min(max_frames, t_break)]
    else:
        clipped = video[:max_frames]
    if clipped.shape[0] < 2:
        return video[: min(max_frames, max(2, video.shape[0]))]
    return clipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess approved clip manifests")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=Path("data/raw_video"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio_extracts"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--allow-synthetic-fallback", action="store_true")
    args = parser.parse_args()

    manifests = _load_manifest_list(args.manifest_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    records: list[dict[str, Any]] = []

    for row in manifests:
        clip_id = str(row["clip_id"])
        seed = abs(hash(clip_id)) % (2**32)
        fallback_video = _synthetic_video(seed) if args.allow_synthetic_fallback else None
        fallback_audio = _synthetic_audio(seed) if args.allow_synthetic_fallback else None

        video_path = args.video_root / f"{clip_id}.npy"
        audio_path = args.audio_root / f"{clip_id}.npy"
        video = _load_or_fallback(video_path, fallback_video)
        audio = _load_or_fallback(audio_path, fallback_audio)
        if video is None or audio is None:
            skipped += 1
            records.append(
                {
                    "clip_id": clip_id,
                    "status": "skipped",
                    "reason": "missing_raw_media",
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                }
            )
            continue
        if video.ndim != 4:
            skipped += 1
            records.append({"clip_id": clip_id, "status": "skipped", "reason": "invalid_video_shape"})
            continue

        video = _crop_before_break(video, row, max_frames=args.max_frames)
        frame_diff = np.diff(video, axis=0, prepend=video[[0]])
        out_file = args.out_dir / f"{clip_id}.npz"
        np.savez_compressed(out_file, video=video.astype(np.float32), audio=audio.astype(np.float32), frame_diff=frame_diff.astype(np.float32))
        processed += 1
        records.append({"clip_id": clip_id, "status": "processed", "out_file": str(out_file)})

    summary = {"processed": processed, "skipped": skipped, "records": records}
    summary_path = args.out_dir / "preprocess_log.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"processed": processed, "skipped": skipped, "log": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
