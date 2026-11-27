"""Self-contained PET-CGDNN (Zhang et al., IEEE Comm. Letters 2021) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class PETCGDNN(nn.Module):
    """Parameter Estimation and Transformation CGDNN for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 75,
        gru_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.phase_estimator = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16 * 16, 1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(2, 8), padding=(0, 4)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, 25, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool1d(4, stride=4) if seq_length >= 512 else nn.Identity()
        self.gru = nn.GRU(25, gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.phase_estimator(x)
        c, s = torch.cos(theta), torch.sin(theta)
        i, q = x[:, 0, :], x[:, 1, :]
        x_rot = torch.stack((i * c + q * s, q * c - i * s), dim=1)
        h = self.conv2(self.conv1(x_rot.unsqueeze(1))).squeeze(2)
        out, _ = self.gru(self.pool(h).transpose(1, 2))
        return self.head(out[:, -1, :])
