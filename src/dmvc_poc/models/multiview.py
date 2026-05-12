"""Multiview baseline models and DMVC-lite fusion."""

from __future__ import annotations

import torch
from torch import nn


class MotionBaseline(nn.Module):
    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 2, 2)),
            nn.Flatten(),
        )
        self.head = nn.Linear(hidden_channels * 4, 1)

    def forward(self, frame_diff: torch.Tensor) -> torch.Tensor:
        x = frame_diff.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.net(x)
        return self.head(feat)


class AudioBaseline(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.head = nn.Linear(hidden * 8, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        feat = self.net(audio)
        return self.head(feat)


class TriModalBaseline(nn.Module):
    def __init__(self, heatmap_grid: int = 8) -> None:
        super().__init__()
        self.rgb_proj = nn.Linear(1, 16)
        self.motion_proj = nn.Linear(1, 16)
        self.audio_proj = nn.Linear(1, 16)
        self.fusion = nn.Sequential(nn.Linear(48, 32), nn.ReLU(inplace=True))
        self.event_head = nn.Linear(32, 1)
        self.heatmap_head = nn.Sequential(nn.Linear(32, heatmap_grid * heatmap_grid), nn.Sigmoid())
        self.heatmap_grid = heatmap_grid

    def forward(self, rgb_logits: torch.Tensor, motion_logits: torch.Tensor, audio_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(
            [
                self.rgb_proj(rgb_logits),
                self.motion_proj(motion_logits),
                self.audio_proj(audio_logits),
            ],
            dim=1,
        )
        feat = self.fusion(x)
        logits = self.event_head(feat)
        heatmap = self.heatmap_head(feat).view(x.shape[0], self.heatmap_grid, self.heatmap_grid)
        return logits, heatmap


class DMVCLiteModel(nn.Module):
    """Simple view-gated DMVC-lite model."""

    def __init__(self, heatmap_grid: int = 8, view_dropout_p: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True), nn.Linear(16, 3))
        self.dropout = nn.Dropout(p=view_dropout_p)
        self.fusion = nn.Sequential(nn.Linear(3, 16), nn.ReLU(inplace=True))
        self.event_head = nn.Linear(16, 1)
        self.time_head = nn.Linear(16, 1)
        self.heatmap_head = nn.Sequential(nn.Linear(16, heatmap_grid * heatmap_grid), nn.Sigmoid())
        self.heatmap_grid = heatmap_grid

    def forward(self, rgb_logits: torch.Tensor, motion_logits: torch.Tensor, audio_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        views = torch.cat([rgb_logits, motion_logits, audio_logits], dim=1)
        gate_logits = self.gate(views)
        weights = torch.softmax(self.dropout(gate_logits), dim=1)
        fused_input = views * weights
        feat = self.fusion(fused_input)
        event_logits = self.event_head(feat)
        time_pred = self.time_head(feat)
        heatmap = self.heatmap_head(feat).view(views.shape[0], self.heatmap_grid, self.heatmap_grid)
        return event_logits, time_pred, heatmap, weights
