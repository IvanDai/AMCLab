"""CNN1: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn

class CNN1(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.features = nn.Sequential(nn.Sequential(nn.Conv2d(1, 50, (1, 8), padding='same'), nn.ReLU()), nn.Dropout(dropout), nn.Sequential(nn.Conv2d(50, 50, (2, 8), padding=0), nn.ReLU()), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(50 * (seq_length - 7), 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        return self.classifier(self.features(x.unsqueeze(1)).flatten(1))
