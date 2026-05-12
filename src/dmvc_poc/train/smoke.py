"""CPU-friendly smoke training used by tests and CI (Phase 0 exit criteria)."""

from __future__ import annotations

import torch
from torch import nn

from dmvc_poc.models.rgb_baseline import RGBTemporalBaseline


def run_smoke_training_step(
    *,
    batch_size: int = 2,
    time_steps: int = 8,
    height: int = 32,
    width: int = 32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    """Run one optimizer step on synthetic tensors; returns loss scalar."""
    torch.manual_seed(seed)
    model = RGBTemporalBaseline(temporal_depth=time_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    video = torch.randn(batch_size, time_steps, 3, height, width, device=device)
    labels = torch.randint(0, 2, (batch_size, 1), device=device).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _heatmap = model(video)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    return {"loss": float(loss.detach().cpu())}
