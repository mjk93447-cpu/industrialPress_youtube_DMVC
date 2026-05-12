"""Minimal RGB temporal baseline aligned with docs/03_model_architecture.md Phase-2 ladder."""

from __future__ import annotations

import torch
from torch import nn


class RGBTemporalBaseline(nn.Module):
    """Lightweight 3D convolution stack over (B, T, 3, H, W) clips."""

    def __init__(
        self,
        *,
        temporal_depth: int = 8,
        hidden_channels: int = 16,
        heatmap_grid: int = 8,
    ) -> None:
        super().__init__()
        self.temporal_depth = temporal_depth
        self.heatmap_grid = heatmap_grid

        self.encoder = nn.Sequential(
            nn.Conv3d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 4, 4)),
            nn.Flatten(),
        )
        flat_dim = hidden_channels * 1 * 4 * 4
        self.precursor_head = nn.Linear(flat_dim, 1)
        self.heatmap_head = nn.Sequential(
            nn.Linear(flat_dim, heatmap_grid * heatmap_grid),
            nn.Sigmoid(),
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            video: Tensor shaped ``(batch, time, 3, height, width)``.

        Returns:
            precursor_logits with shape ``(batch, 1)`` and heatmap with shape
            ``(batch, heatmap_grid, heatmap_grid)``.
        """
        if video.dim() != 5:
            msg = f"Expected video rank 5, got shape {tuple(video.shape)}"
            raise ValueError(msg)
        b, t, c, h, w = video.shape
        if c != 3:
            msg = f"Expected 3 RGB channels, got {c}"
            raise ValueError(msg)
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.encoder(x)
        logits = self.precursor_head(feat)
        heatmap = self.heatmap_head(feat).view(b, self.heatmap_grid, self.heatmap_grid)
        return logits, heatmap
