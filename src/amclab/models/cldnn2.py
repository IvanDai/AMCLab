"""CLDNN2: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn

class CLDNN2(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        layers = []
        for a, b, k in ((1, 256, (1, 3)), (256, 256, (2, 3)), (256, 80, (1, 3)), (80, 80, (1, 3))):
            layers.extend((nn.Sequential(nn.Conv2d(a, b, k, padding=0), nn.ReLU()), nn.Dropout(dropout)))
        self.features = nn.Sequential(*layers)
        self.lstm = nn.LSTM(seq_length - 8, 50, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        h = self.features(x.unsqueeze(1))
        h = h.flatten(1).reshape(x.shape[0], 80, self.seq_length - 8)
        h, _ = self.lstm(h)
        return self.classifier(h[:, -1])
