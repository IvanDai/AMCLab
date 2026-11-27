"""Self-contained CGDNet (Njoku et al., 2021 / AMR-Benchmark) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class CGDNet(nn.Module):
    """Convolutional + GRU + Deep Neural Network for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 50,
        gru_hidden: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(1, 6), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 6), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(2, 6), padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(channels, gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block3(self.block2(self.block1(x.unsqueeze(1)))).squeeze(2)
        out, _ = self.gru(h.transpose(1, 2))
        return self.head(out[:, -1, :])
