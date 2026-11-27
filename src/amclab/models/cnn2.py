"""Self-contained CNN2 (4-stage 2D CNN, AMR-Benchmark) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNN2(nn.Module):
    """4-stage 2D CNN with MaxPool downsampling for variable IQ length ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(c, c, kernel_size=(2, 8), padding=(0, 4)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(c, c, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(c, c, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.unsqueeze(1)))
