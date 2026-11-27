"""CLDNN: native PyTorch architecture port of AMR-Benchmark."""
import torch
from torch import nn

class CLDNN(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)

        def block(cin):
            return nn.Sequential(nn.ZeroPad2d((2, 2, 0, 0)), nn.Sequential(nn.Conv2d(cin, 50, (1, 8), padding=0), nn.ReLU()), nn.Dropout(dropout))
        self.conv1, self.conv2, self.conv3 = (block(1), block(50), block(50))
        self.lstm = nn.LSTM(4 * seq_length - 24, 50, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(50, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        a = self.conv1(x.unsqueeze(1))
        c = self.conv3(self.conv2(a))
        h = torch.cat((a, c), dim=-1).reshape(x.shape[0], 50, -1)
        h, _ = self.lstm(h)
        return self.classifier(h[:, -1])
