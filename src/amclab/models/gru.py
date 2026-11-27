"""Self-contained GRU (Rajendran et al., 2017) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRU(nn.Module):
    """2-layer GRU modulation classifier for variable IQ length ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4) if seq_length >= 512 else nn.Identity()
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.downsample(x).transpose(1, 2))
        return self.head(out[:, -1, :])
