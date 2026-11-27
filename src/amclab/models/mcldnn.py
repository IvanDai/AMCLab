"""MCLDNN: multi-channel convolution and two-layer LSTM.

Uses native PyTorch layout/reshape as in IQFormer/model/MCLDNN.py.
Network topology follows the AMR-Benchmark baseline.
"""
import torch
from torch import nn

class MCLDNN(nn.Module):
    def __init__(self, num_classes=11, seq_length=128, dropout=.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = seq_length

        self.conv1 = nn.Sequential(nn.Conv2d(1, 50, (2, 8), padding="same"), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ConstantPad1d((7, 0), 0), nn.Conv1d(1, 50, 8), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ConstantPad1d((7, 0), 0), nn.Conv1d(1, 50, 8), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(50, 50, (1, 8), padding="same"), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(100, 100, (2, 5)), nn.ReLU())
        self.lstm = nn.LSTM(100, 128, batch_first=True, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        a = self.conv1(x)
        b = self.conv2(x[:, :, 0, :])
        c = self.conv3(x[:, :, 1, :])
        h = self.conv5(torch.cat((a, self.conv4(torch.stack((b, c), dim=2))), dim=1))
        h = h.reshape(x.shape[0], self.seq_length - 4, 100)
        y, _ = self.lstm(h)
        return self.classifier(y[:, -1, :])
