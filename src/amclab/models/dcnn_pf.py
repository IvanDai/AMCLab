"""1DCNN-PF: two streams and five convolution/pooling fusion stages."""
import math
import torch
from torch import nn

class DCNNPF(nn.Module):
    def __init__(self, num_classes=11, seq_length=128, dropout=.5, input_mode="iq"):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        if input_mode not in ("iq", "ap", "iq_to_ap"):
            raise ValueError("input_mode must be iq, ap (preconverted), or iq_to_ap.")
        self.seq_length = seq_length
        self.input_mode = input_mode

        def block(in_channels):
            return nn.Sequential(nn.Conv1d(in_channels, 64, 3), nn.ReLU(), nn.Dropout(.2))

        def stream():
            return nn.Sequential(*(block(a) for a in (1, 64, 64, 64)))

        self.stream1, self.stream2 = stream(), stream()
        self.fusion = nn.Sequential(*(
            m for a in (128, 64, 64, 64, 64)
            for m in (block(a), nn.MaxPool1d(2))
        ))
        n = seq_length - 8
        for _ in range(5):
            n = (n - 2) // 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*n, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        if self.input_mode == "iq_to_ap":
            i, q = (x.float() if x.dtype in (torch.float16, torch.bfloat16) else x).unbind(dim=1)
            x = torch.stack((torch.hypot(i, q), torch.atan2(q, i)/math.pi), dim=1).to(x.dtype)
        h = self.fusion(torch.cat((self.stream1(x[:, 0:1]), self.stream2(x[:, 1:2])), dim=1))
        return self.classifier(h)
