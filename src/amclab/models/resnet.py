"""RESNET: native PyTorch architecture port of AMR-Benchmark."""
from torch import nn

class ResNet(nn.Module):

    def __init__(self, num_classes=11, seq_length=128, dropout=0.6):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 256, (1, 3), padding='same'), nn.ReLU())
        self.conv2 = nn.Conv2d(256, 256, (2, 3), padding='same')
        self.conv3 = nn.Sequential(nn.Conv2d(256, 80, (1, 3), padding='same'), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(80, 80, (1, 3), padding='same'), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(80 * 2 * seq_length, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        x = x.unsqueeze(1)
        y = (self.conv2(self.conv1(x)) + x).relu()
        return self.classifier(self.drop(self.conv4(self.conv3(y))).flatten(1))

class _ResidualStack1D(nn.Module):

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False)

        def branch():
            return nn.Sequential(nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True), nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm1d(out_ch))
        self.res1, self.res2 = (branch(), branch())
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        h = self.proj(x)
        h = self.act(h + self.res1(h))
        return self.pool(self.act(h + self.res2(h)))

class LegacyResNet(nn.Module):
    """Unmodified architecture of AMCLab's previous ResNet size presets."""

    def __init__(self, num_classes=11, seq_length=128, channels=48, num_stacks=4, dropout=0.3):
        super().__init__()
        if seq_length not in (128, 1024):
            raise ValueError('seq_length must be 128 or 1024.')
        self.seq_length = int(seq_length)
        self.backbone = nn.Sequential(*[_ResidualStack1D(2 if i == 0 else channels, channels, 2 if i < num_stacks - 1 or seq_length >= 512 else 1) for i in range(num_stacks)])
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4, 128), nn.SELU(inplace=True), nn.AlphaDropout(dropout), nn.Linear(128, 128), nn.SELU(inplace=True), nn.AlphaDropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f'Expected [B,2,{self.seq_length}], got {tuple(x.shape)}')
        return self.classifier(self.pool(self.backbone(x)))

def resnet_small(num_classes=11, seq_length=128, **kwargs):
    return LegacyResNet(num_classes, seq_length, channels=32, num_stacks=3, **kwargs)

def resnet_base(num_classes=11, seq_length=128, **kwargs):
    return LegacyResNet(num_classes, seq_length, channels=48, num_stacks=4, **kwargs)

def resnet_large(num_classes=11, seq_length=128, **kwargs):
    return LegacyResNet(num_classes, seq_length, channels=64, num_stacks=6, **kwargs)
