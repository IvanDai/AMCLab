"""Self-contained CLDNN (West & O'Shea, 2017) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLDNN(nn.Module):
    """Convolutional Long Short-Term Deep Neural Network for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 50,
        lstm_hidden: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(1, 8), padding=(0, 4))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 8), padding=(0, 4))
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=(2, 8), padding=(0, 4))
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4) if seq_length >= 512 else nn.Identity()
        self.lstm = nn.LSTM(channels * 2, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.conv1(x.unsqueeze(1)))
        h3 = F.relu(self.conv3(F.relu(self.conv2(h1))))
        min_l = min(h1.shape[-1], h3.shape[-1])
        feat = self.pool(torch.cat([h1[:, :, 0, :min_l], h3[:, :, 0, :min_l]], dim=1))
        out, _ = self.lstm(feat.transpose(1, 2))
        return self.classifier(out[:, -1, :])
