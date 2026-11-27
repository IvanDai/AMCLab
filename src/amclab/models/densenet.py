"""DENSENET: native PyTorch architecture port of AMR-Benchmark."""
import torch
from torch import nn

class DenseNet(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.6):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, (1, 3), padding='same'), nn.ReLU())
        self.conv2 = nn.Conv2d(256, 256, (2, 3), padding='same')
        self.conv3 = nn.Conv2d(512, 80, (1, 3), padding='same')
        self.conv4 = nn.Sequential(nn.Conv2d(592, 80, (1, 3), padding='same'), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(80 * 2 * seq_length, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        a = self.conv1(x.unsqueeze(1))
        b = self.conv2(a)
        c = self.conv3(torch.cat((a, b), dim=1).relu())
        d = self.conv4(torch.cat((a, b, c), dim=1).relu())
        return self.classifier(self.drop(d).flatten(1))
