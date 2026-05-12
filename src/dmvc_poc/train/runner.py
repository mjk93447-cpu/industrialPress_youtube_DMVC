"""Training helpers for baseline ladder and DMVC-lite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dmvc_poc.models import AudioBaseline, DMVCLiteModel, MotionBaseline, RGBTemporalBaseline, TriModalBaseline


@dataclass
class TrainResult:
    mode: str
    train_loss: float
    val_loss: float
    checkpoint_path: Path
    log_path: Path


def _load_npz_rows(processed_dir: Path) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    for path in sorted(processed_dir.glob("*.npz")):
        with np.load(path) as data:
            rows.append(
                {
                    "video": data["video"].astype(np.float32),
                    "audio": data["audio"].astype(np.float32),
                    "frame_diff": data["frame_diff"].astype(np.float32),
                }
            )
    return rows


def _stack_batch(rows: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    min_t = min(r["video"].shape[0] for r in rows)
    min_audio = min(r["audio"].shape[0] for r in rows)
    min_h = min(r["video"].shape[2] for r in rows)
    min_w = min(r["video"].shape[3] for r in rows)
    videos = np.stack([r["video"][:min_t, :, :min_h, :min_w] for r in rows], axis=0)
    diffs = np.stack([r["frame_diff"][:min_t, :, :min_h, :min_w] for r in rows], axis=0)
    audios = np.stack([r["audio"][:min_audio] for r in rows], axis=0)
    labels = np.array([[1.0 if i % 2 == 0 else 0.0] for i in range(len(rows))], dtype=np.float32)
    time_target = np.array([[0.5 + 0.05 * i] for i in range(len(rows))], dtype=np.float32)
    return {
        "video": torch.from_numpy(videos).to(device),
        "frame_diff": torch.from_numpy(diffs).to(device),
        "audio": torch.from_numpy(audios).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "time_target": torch.from_numpy(time_target).to(device),
    }


def _make_models(mode: str, time_steps: int) -> dict[str, nn.Module]:
    rgb = RGBTemporalBaseline(temporal_depth=time_steps)
    motion = MotionBaseline()
    audio = AudioBaseline()
    tri = TriModalBaseline()
    dmvc = DMVCLiteModel()
    if mode == "rgb_temporal_baseline":
        return {"rgb": rgb}
    if mode == "rgb_motion_baseline":
        return {"rgb": rgb, "motion": motion}
    if mode == "rgb_audio_baseline":
        return {"rgb": rgb, "audio": audio}
    if mode == "tri_modal_baseline":
        return {"rgb": rgb, "motion": motion, "audio": audio, "tri": tri}
    if mode == "dmvc_lite":
        return {"rgb": rgb, "motion": motion, "audio": audio, "dmvc": dmvc}
    msg = f"Unknown train mode: {mode}"
    raise ValueError(msg)


def run_training(
    *,
    mode: str,
    processed_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 0,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_npz_rows(processed_dir)
    if not rows:
        msg = f"No preprocessed .npz files found in {processed_dir}"
        raise FileNotFoundError(msg)
    split = max(1, int(0.8 * len(rows)))
    train_rows = rows[:split]
    val_rows = rows[split:] if len(rows) > split else rows[:1]

    device = torch.device("cpu")
    batch_train = _stack_batch(train_rows, device)
    batch_val = _stack_batch(val_rows, device)
    models = _make_models(mode, time_steps=batch_train["video"].shape[1])
    for m in models.values():
        m.to(device)
    params = [p for m in models.values() for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    train_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        rgb_logits, _rgb_heatmap = models["rgb"](batch_train["video"])
        loss = bce(rgb_logits, batch_train["labels"])
        if "motion" in models:
            motion_logits = models["motion"](batch_train["frame_diff"])
            loss = loss + 0.5 * bce(motion_logits, batch_train["labels"])
        else:
            motion_logits = rgb_logits.detach()
        if "audio" in models:
            audio_logits = models["audio"](batch_train["audio"])
            loss = loss + 0.5 * bce(audio_logits, batch_train["labels"])
        else:
            audio_logits = rgb_logits.detach()

        if "tri" in models:
            fused_logits, _heatmap = models["tri"](rgb_logits, motion_logits, audio_logits)
            loss = loss + 0.5 * bce(fused_logits, batch_train["labels"])

        if "dmvc" in models:
            dmvc_logits, time_pred, _heatmap, _weights = models["dmvc"](rgb_logits, motion_logits, audio_logits)
            loss = loss + 0.5 * bce(dmvc_logits, batch_train["labels"]) + 0.2 * mse(time_pred, batch_train["time_target"])
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

    with torch.no_grad():
        rgb_logits_v, _ = models["rgb"](batch_val["video"])
        val_loss_t = bce(rgb_logits_v, batch_val["labels"])
        val_loss = float(val_loss_t.detach().cpu())

    checkpoint_path = output_dir / f"{mode}.pt"
    payload: dict[str, Any] = {"mode": mode, "models": {k: v.state_dict() for k, v in models.items()}}
    torch.save(payload, checkpoint_path)
    log_path = output_dir / f"{mode}_train_log.json"
    log_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainResult(mode=mode, train_loss=train_loss, val_loss=val_loss, checkpoint_path=checkpoint_path, log_path=log_path)
