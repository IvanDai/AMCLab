"""Self-contained DenseNet (Liu et al., 2017 / AMR-Benchmark) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Densely Connected 1D CNN baseline for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        growth_rate: int = 32,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, growth_rate, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList()
        ch = growth_rate
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(ch, growth_rate, kernel_size=3, padding=1, bias=False),
                )
            )
            ch += growth_rate
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for layer in self.layers:
            h = torch.cat([h, layer(h)], dim=1)
        return self.classifier(self.pool(h))
