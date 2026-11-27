"""Self-contained DAE (Ke & Vikalo, IEEE TWC 2021) baseline classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class DAE(nn.Module):
    """LSTM-based Denoising Autoencoder classifier returning logits ``[B, num_classes]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        hidden_size: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4) if seq_length >= 512 else nn.Identity()
        self.encoder1 = nn.LSTM(2, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.encoder2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder1(self.downsample(x).transpose(1, 2))
        _, (h_n, _) = self.encoder2(self.drop(h))
        return self.classifier(h_n[-1])
