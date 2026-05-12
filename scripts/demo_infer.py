#!/usr/bin/env python3
"""Inference demo: pre-failure probability timeline + damage heatmap image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from dmvc_poc.infer import normalize_to_uint8


def _heatmap_from_clip(video: np.ndarray, frame_diff: np.ndarray) -> np.ndarray:
    # Proxy latent: combine deformation magnitude + texture intensity
    h1 = np.mean(np.abs(frame_diff), axis=(0, 1))
    h2 = np.mean(np.abs(video), axis=(0, 1))
    heat = 0.7 * h1 + 0.3 * h2
    return heat[:8, :8]


def _timeline(video: np.ndarray, end_sec: float, fps: float) -> dict[str, list[float]]:
    t = video.shape[0]
    cutoff = min(t, max(2, int(end_sec * fps)))
    x = np.linspace(-2.5, 2.5, cutoff)
    p = 1.0 / (1.0 + np.exp(-x))
    ttb = np.maximum(0.0, (cutoff - np.arange(cutoff)) / fps)
    return {"pre_failure_probability": p.tolist(), "time_to_break_seconds": ttb.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run demo inference from preprocessed clip")
    parser.add_argument("--clip", required=True, help="clip_id or path to .npz")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--end-sec", type=float, default=0.8)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()

    clip_path = Path(args.clip)
    if clip_path.suffix != ".npz":
        clip_path = args.processed_dir / f"{args.clip}.npz"
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    with np.load(clip_path) as data:
        video = data["video"]
        frame_diff = data["frame_diff"]

    heat = _heatmap_from_clip(video, frame_diff)
    latent_path = None
    timeline = _timeline(video, end_sec=args.end_sec, fps=args.fps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    heat_img = Image.fromarray(normalize_to_uint8(heat)).resize((256, 256), resample=Image.Resampling.NEAREST)
    heat_path = args.out_dir / f"{clip_path.stem}_heatmap.png"
    heat_img.save(heat_path)
    latent_path = args.out_dir / f"{clip_path.stem}_latent.npy"
    np.save(latent_path, heat.astype(np.float32))

    # Side-by-side card using first frame + heatmap for quick inspection
    frame0 = video[0]
    frame0_img = np.transpose(frame0, (1, 2, 0))
    frame0_u8 = normalize_to_uint8(frame0_img)
    frame0_pil = Image.fromarray(frame0_u8).resize((256, 256), resample=Image.Resampling.NEAREST)
    card = Image.new("RGB", (512, 256))
    card.paste(frame0_pil.convert("RGB"), (0, 0))
    card.paste(heat_img.convert("RGB"), (256, 0))
    card_path = args.out_dir / f"{clip_path.stem}_card.png"
    card.save(card_path)

    json_path = args.out_dir / f"{clip_path.stem}_timeline.json"
    json_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "heatmap_png": str(heat_path),
                "timeline_json": str(json_path),
                "card_png": str(card_path),
                "latent_npy": str(latent_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
