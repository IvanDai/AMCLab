"""MCNet: six M-blocks and length-dependent terminal pooling.

Fixed asymmetric padding is written with native PyTorch layers. All
strided stages have even input widths for L=128 and L=1024.
"""
import torch
from torch import nn

class _MBlock(nn.Module):
    def __init__(self, cin, down=False, final=False, prepool=False):
        super().__init__()
        self.prepool = (
            nn.MaxPool2d((1, 3), stride=(1, 2), ceil_mode=True)
            if prepool else nn.Identity()
        )
        self.reduce = nn.Sequential(nn.Conv2d(cin, 32, 1), nn.ReLU())
        a, b = (96, 64) if final else (48, 32)
        stride = (1, 2) if down else (1, 1)
        self.vertical = nn.Sequential(
            nn.Conv2d(32, a, (3, 1), padding=(1, 0)), nn.ReLU()
        )
        self.vpool = (
            nn.MaxPool2d((1, 3), stride=(1, 2), ceil_mode=True)
            if down else nn.Identity()
        )
        self.horizontal = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 0)) if down else nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(32, a, (1, 3), stride=stride), nn.ReLU()
        )
        self.short = nn.Sequential(
            nn.Conv2d(32, b, 1, stride=stride), nn.ReLU()
        )

    def forward(self, x):
        x = self.reduce(self.prepool(x))
        return torch.cat((self.short(x), self.vpool(self.vertical(x)), self.horizontal(x)), dim=1)


class MCNet(nn.Module):
    def __init__(self, num_classes=11, seq_length=128, dropout=.5):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError("seq_length must be 128 or 1024.")
        self.seq_length = seq_length

        self.stem = nn.Sequential(
            nn.ZeroPad2d((2, 3, 1, 1)),
            nn.Conv2d(1, 64, (3, 7), stride=(1, 2)), nn.ReLU(),
            nn.MaxPool2d((1, 3), stride=(1, 2), ceil_mode=True),
        )
        self.pre1 = nn.Sequential(
            nn.Conv2d(64, 32, (3, 1), padding=(1, 0)), nn.ReLU(),
            nn.AvgPool2d((1, 3), stride=(1, 2), ceil_mode=True, count_include_pad=False),
        )
        self.pre2 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 0)),
            nn.Conv2d(64, 32, (1, 3), stride=(1, 2)), nn.ReLU(),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=(1, 2)), nn.ReLU(),
            nn.MaxPool2d((1, 3), stride=(1, 2), ceil_mode=True),
        )
        self.m1 = _MBlock(64, down=True, prepool=True)
        self.m2 = _MBlock(128)
        self.m3 = _MBlock(128, down=True)
        self.m4 = _MBlock(128)
        self.m5 = _MBlock(128, down=True)
        self.m6 = _MBlock(128, final=True)
        # Skip inputs are non-negative after ReLU and residual addition.
        self.skip_pool = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.MaxPool2d((2, 2), stride=(1, 2)),
        )
        self.last_pool = nn.AvgPool2d((2, 1 if seq_length == 128 else 8))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Flatten(), nn.Linear(384, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected [B,2,{self.seq_length}], got {tuple(x.shape)}")
        x = self.stem(x.unsqueeze(1))
        x = torch.cat((self.pre1(x), self.pre2(x)), dim=1)
        x = self.m1(x) + self.skip(x)
        x = self.m2(x) + x
        x = self.m3(x) + self.skip_pool(x)
        x = self.m4(x) + x
        x = self.m5(x) + self.skip_pool(x)
        x = torch.cat((self.m6(x), x), dim=1)
        return self.classifier(self.last_pool(x))
