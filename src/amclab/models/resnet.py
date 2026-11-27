"""Self-contained ResNet (O'Shea et al., IEEE JSTSP 2018) with small/base/large variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualStack1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.res1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.res2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.act(h + self.res1(h))
        h = self.act(h + self.res2(h))
        return self.pool(h)


class ResNet(nn.Module):
    """Deep Residual 1D CNN for modulation classification ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 48,
        num_stacks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        stacks = []
        in_ch = 2
        for i in range(num_stacks):
            stride = 2 if (i < num_stacks - 1 or seq_length >= 512) else 1
            stacks.append(_ResidualStack1D(in_ch, channels, stride=stride))
            in_ch = channels
        self.backbone = nn.Sequential(*stacks)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.backbone(x)))


def resnet_small(num_classes: int = 11, seq_length: int = 128, **kwargs) -> ResNet:
    return ResNet(num_classes=num_classes, seq_length=seq_length, channels=32, num_stacks=3, **kwargs)


def resnet_base(num_classes: int = 11, seq_length: int = 128, **kwargs) -> ResNet:
    return ResNet(num_classes=num_classes, seq_length=seq_length, channels=48, num_stacks=4, **kwargs)


def resnet_large(num_classes: int = 11, seq_length: int = 128, **kwargs) -> ResNet:
    return ResNet(num_classes=num_classes, seq_length=seq_length, channels=64, num_stacks=6, **kwargs)
