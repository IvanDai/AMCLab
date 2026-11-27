"""IC_AMCNET: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn
import torch

class GaussianNoise(nn.Module):

    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std if self.training else x

class ICAMCNet(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.4, noise_std=1.0):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.features = nn.Sequential(nn.Sequential(nn.Conv2d(1, 64, (1, 8), padding='same'), nn.ReLU()), nn.MaxPool2d((2, 2)), nn.Sequential(nn.Conv2d(64, 64, (1, 4), padding='same'), nn.ReLU()), nn.Sequential(nn.Conv2d(64, 128, (1, 8), padding='same'), nn.ReLU()), nn.MaxPool2d((1, 1)), nn.Dropout(dropout), nn.Sequential(nn.Conv2d(128, 128, (1, 8), padding='same'), nn.ReLU()), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(128 * (seq_length // 2), 128), nn.ReLU(), nn.Dropout(dropout), GaussianNoise(noise_std), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        return self.classifier(self.features(x.unsqueeze(1)).flatten(1))
