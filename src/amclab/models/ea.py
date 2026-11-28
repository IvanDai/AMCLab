"""Expert-Assistant (E-A RAW), PyTorch port for AMCLab.
Based on supplied Keras expert_assistant.py, not E-A (BEST).
He et al., DOI: 10.1109/TWC.2025.3645099.
Input [B,2,L], output [B,num_classes] logits.
"""
import torch
from torch import nn
from torch.nn import functional as F


def temporal_shuffle(x):
    """Interleave assistant channels as in the Keras NHWC reshape."""
    b, c, h, t = x.shape
    if c != 24:
        raise ValueError("Expected 24 assistant channels.")
    return x.reshape(b, 2, 12, h, t).transpose(1, 2).reshape(b, c, h, t)


class ExpertAssistant(nn.Module):
    def __init__(self, num_classes: int = 11, seq_length: int = 128):
        super().__init__()
        self.num_classes = int(num_classes)
        self.seq_length = int(seq_length)
        if self.num_classes < 1 or self.seq_length < 2:
            raise ValueError("Invalid class count or sequence length.")
        self.conv_b1 = nn.Conv2d(1, 75, (2, 8))
        self.conv_b2 = nn.Conv2d(75, 24, (1, 5), padding=(0, 2))
        self.pool = nn.AvgPool2d((1, 2))
        self.conv1_a1out = nn.Conv2d(1, 12, (2, 8))
        self.conv1_a2out = nn.Conv2d(1, 12, (2, 8))
        self.conv2_a1out = nn.Conv2d(12, 4, (1, 5), padding=(0, 2))
        self.conv2_a2out = nn.Conv2d(12, 4, (1, 5), padding=(0, 2))
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.dense_class = nn.Linear(64, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        # Keras initialization families; random draws differ across frameworks.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

    @staticmethod
    def _first_conv(x, conv):
        return F.relu(conv(F.pad(x, (3, 4, 0, 0))))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected IQ [B,2,{self.seq_length}], got {tuple(x.shape)}")
        # Original 128 unchanged; pad odd lengths to a multiple of two.
        if self.seq_length % 2:
            x = F.pad(x, (0, 1))
        x = x.unsqueeze(1)
        expert = self.pool(F.relu(self.conv_b2(self._first_conv(x, self.conv_b1))))
        first, second = x.chunk(2, dim=-1)
        a = self._first_conv(first, self.conv1_a1out)
        b = self._first_conv(second, self.conv1_a2out)
        a, b = temporal_shuffle(torch.cat((a, b), dim=1)).chunk(2, dim=1)
        a = F.relu(self.conv2_a1out(a))
        b = F.relu(self.conv2_a2out(b))
        features = torch.cat((expert, a, b), dim=1).squeeze(2).transpose(1, 2)
        _, h = self.gru(features.contiguous())
        return self.dense_class(h[-1])


EA = ExpertAssistant
Expert_Assistant = ExpertAssistant
EANet = ExpertAssistant

if __name__ == "__main__":
    model = ExpertAssistant()
    y = model(torch.randn(2, 2, 128))
    y.square().mean().backward()
    print(y.shape, sum(p.numel() for p in model.parameters()))
