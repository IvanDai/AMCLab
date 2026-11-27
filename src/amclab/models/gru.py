"""GRU: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn

class GRU(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, hidden_size=128):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.rnn = nn.GRU(2, hidden_size, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        y, _ = self.rnn(x.transpose(1, 2))
        return self.classifier(y[:, -1, :])
