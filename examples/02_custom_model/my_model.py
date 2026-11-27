"""User-defined custom AMC network with lightweight and deep variants."""

import torch
import torch.nn as nn


class MyCustomAMCNet(nn.Module):
    """Custom 1D ConvNet following AMCLab's pure IQ contract: [B, 2, L] -> [B, C]."""

    def __init__(
        self,
        num_classes: int,
        width: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(2, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        ]
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(width, width, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout1d(dropout),
            ])
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(x)).flatten(1))
