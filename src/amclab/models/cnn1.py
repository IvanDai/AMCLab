"""Self-contained CNN1 (VT-CNN2, O'Shea et al., 2016) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN1(nn.Module):
    """VT-CNN2 adapted for variable IQ sequence lengths ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        c2 = max(32, channels // 3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, c2, kernel_size=(2, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(self.conv1(x.unsqueeze(1)))
        return self.classifier(self.pool(h))
