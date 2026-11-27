"""Self-contained MCNet (Huynh-The et al., IEEE TWC 2020) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MBlock(nn.Module):
    """Multi-scale asymmetric parallel convolution block used in MCNet."""

    def __init__(self, in_ch: int, out_ch: int, use_pool: bool = True):
        super().__init__()
        c_branch = out_ch // 3
        c_last = out_ch - 2 * c_branch
        self.b1 = nn.Conv2d(in_ch, c_branch, kernel_size=(1, 1))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c_branch, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_branch, c_branch, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c_last, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_last, c_last, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return F.relu(self.pool(y), inplace=True)


class MCNet(nn.Module):
    """MCNet with multi-scale skip blocks for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        base_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.m1 = _MBlock(c, c, use_pool=True)
        self.m2 = _MBlock(c, c, use_pool=False)
        self.m3 = _MBlock(c, c, use_pool=True)
        self.m4 = _MBlock(c, c, use_pool=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.unsqueeze(1))
        h = self.m1(h)
        h = h + self.m2(h)
        h = self.m3(h)
        h = h + self.m4(h)
        return self.head(self.pool(h))
