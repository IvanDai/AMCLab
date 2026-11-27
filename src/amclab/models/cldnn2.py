"""Self-contained CLDNN2 (Liu et al., 2017 / AMR-Benchmark) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class CLDNN2(nn.Module):
    """Extended 3-conv + LSTM CLDNN2 architecture for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 64,
        lstm_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(1, 8), padding=(0, 4)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(channels, channels, kernel_size=(2, 8), padding=(0, 4)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(channels, channels, kernel_size=(1, 4), padding=(0, 2)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(channels, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs(x.unsqueeze(1)).squeeze(2)
        out, _ = self.lstm(h.transpose(1, 2))
        return self.classifier(out[:, -1, :])
