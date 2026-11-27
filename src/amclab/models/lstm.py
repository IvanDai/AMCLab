"""LSTM and LSTM-AP: identical recurrent architecture, explicit input representation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    """Use raw IQ at both lengths; no implicit length-dependent preprocessing."""
    def __init__(self, num_classes=11, seq_length=128, hidden_size=128):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = seq_length
        self.rnn = nn.LSTM(2, hidden_size, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        y, _ = self.rnn(x.transpose(1, 2))
        return self.classifier(y[:, -1, :])


class LSTMAP(LSTM):
    """Accept raw IQ; convert to amplitude and phase/pi for either length.

    normalize_amplitude=True optionally applies per-sample L2
    normalization, as used in the original RML2016 LSTM preprocessing.
    """
    def __init__(self, num_classes=11, seq_length=128, hidden_size=128,
                 normalize_amplitude=False):
        super().__init__(num_classes, seq_length, hidden_size)
        self.normalize_amplitude = normalize_amplitude

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        # Use float32 for amplitude/phase arithmetic with half inputs.
        i, q = (x.float() if x.dtype in (torch.float16, torch.bfloat16) else x).unbind(dim=1)
        amplitude = torch.hypot(i, q)
        if self.normalize_amplitude:
            amplitude = F.normalize(amplitude, p=2, dim=-1)
        phase = torch.atan2(q, i) / math.pi
        return super().forward(torch.stack((amplitude, phase), dim=1).to(x.dtype))
