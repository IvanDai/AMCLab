"""Self-contained IC-AMCNet (Hermawan et al., IEEE Access 2020) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class ICAMCNet(nn.Module):
    """IC-AMCNet with Gaussian noise regularization for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 64,
        dropout: float = 0.4,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.noise_std = float(noise_std)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=(2, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv4(self.conv3(self.conv2(self.conv1(x.unsqueeze(1)))))
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        return self.classifier(h)
