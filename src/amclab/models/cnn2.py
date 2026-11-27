"""CNN2: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn

class CNN2(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        blocks = []
        for cin, cout in zip((1, 256, 128, 64), (256, 128, 64, 64)):
            blocks.extend((nn.Sequential(nn.Conv2d(cin, cout, (2, 8), padding='same'), nn.ReLU()), nn.MaxPool2d((1, 2)), nn.Dropout(dropout)))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Linear(64 * 2 * (seq_length // 16), 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        return self.classifier(self.features(x.unsqueeze(1)).flatten(1))
