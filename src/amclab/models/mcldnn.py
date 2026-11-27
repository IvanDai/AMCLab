"""Self-contained MCLDNN (Xu et al., IEEE TCCN 2020) baseline model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLDNN(nn.Module):
    """Multi-Channel Convolutional Long Short-Term Deep Neural Network for ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        channels: int = 50,
        lstm_hidden: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv_iq = nn.Conv2d(1, channels, kernel_size=(2, 8), padding=(1, 3))
        self.conv_i = nn.Conv1d(1, channels, kernel_size=8, padding=3)
        self.conv_q = nn.Conv1d(1, channels, kernel_size=8, padding=3)
        self.conv_merge = nn.Conv2d(channels, channels, kernel_size=(2, 5), padding=(0, 2))
        self.conv_fuse = nn.Conv2d(channels * 2, channels * 2, kernel_size=(2, 5), padding=(0, 2))
        self.pool = nn.AvgPool1d(4, stride=4) if seq_length >= 512 else nn.Identity()
        self.lstm1 = nn.LSTM(channels * 2, lstm_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        x_iq = F.relu(self.conv_iq(x.unsqueeze(1)))[:, :, :2, :length]
        x_i = F.relu(self.conv_i(x[:, 0:1, :]))[:, :, :length]
        x_q = F.relu(self.conv_q(x[:, 1:2, :]))[:, :, :length]
        x_sep = F.relu(self.conv_merge(torch.stack([x_i, x_q], dim=2))).expand(-1, -1, 2, -1)

        fused = F.relu(self.conv_fuse(torch.cat([x_iq, x_sep], dim=1))).squeeze(2)
        fused = self.pool(fused).transpose(1, 2)
        out, _ = self.lstm1(fused)
        out, _ = self.lstm2(out)
        return self.classifier(out[:, -1, :])
