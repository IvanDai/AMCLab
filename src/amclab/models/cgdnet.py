"""CGDNET: native PyTorch architecture port of AMR-Benchmark."""
import math
import torch
from torch import nn

class GaussianDropout(nn.Module):

    def __init__(self, p=0.2):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError('GaussianDropout p must be in [0,1).')
        self.std = math.sqrt(p / (1 - p))

    def forward(self, x):
        return x * (1 + torch.randn_like(x) * self.std) if self.training and self.std else x

class CGDNet(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, channels=50, gru_hidden=50, dropout=0.2):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)

        def block(a, b):
            return nn.Sequential(nn.Sequential(nn.Conv2d(a, b, (1, 6), padding=0), nn.ReLU()), nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d((2, 2), stride=1), GaussianDropout(dropout))
        self.block1 = block(1, channels)
        self.block2 = block(channels, channels)
        self.block3 = block(channels, channels)
        self.gru = nn.GRU(4 * seq_length - 40, gru_hidden, batch_first=True)
        self.head = nn.Sequential(GaussianDropout(dropout), nn.Linear(gru_hidden, 256), nn.ReLU(), GaussianDropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        a = self.block1(x.unsqueeze(1))
        c = self.block3(self.block2(a))
        h = torch.cat((a, c), dim=-1).reshape(x.shape[0], a.shape[1], -1)
        y, _ = self.gru(h)
        return self.head(y[:, -1])
