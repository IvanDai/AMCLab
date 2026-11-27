"""PET_CGDNN: native PyTorch architecture port of AMR-Benchmark."""
import torch
from torch import nn

class PET(nn.Module):

    def __init__(self, frame_length=128):
        super().__init__()
        self.p1 = nn.Sequential(nn.Flatten(), nn.Linear(frame_length * 2, 1))

    def forward(self, x):
        theta = self.p1(x)
        i, q = (x[:, :, 0], x[:, :, 1])
        c, s = (theta.cos(), theta.sin())
        return torch.stack((i * c + q * s, q * c - i * s), dim=1).unsqueeze(1)

class PETCGDNN(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, hidden_size=128):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.features = nn.Sequential(PET(seq_length), nn.Conv2d(1, 75, (2, 8)), nn.ReLU(inplace=True), nn.Conv2d(75, 25, (1, 5)), nn.ReLU(inplace=True))
        self.gru = nn.GRU(25, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        h = self.features(x.transpose(1, 2))
        h = h.squeeze(2).transpose(1, 2)
        y, _ = self.gru(h)
        return self.classifier(y[:, -1, :])
