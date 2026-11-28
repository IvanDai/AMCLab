"""AMCLab-compatible port of the official zjwfufu/AMC-Net implementation.

The network operations intentionally follow the authors' official
models/model.py, including its original ACM, FFM, normalization, and
classifier. Only the constructor/interface and input validation are adapted
for AMCLab.

Input:  [B, 2, L] (I/Q)
Output: [B, num_classes] logits

Official source:
https://github.com/zjwfufu/AMC-Net/blob/main/models/model.py
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_c = in_channel
        self.out_c = out_channel
        self.conv_block = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(self.in_c, self.out_c, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c),
        )

    def forward(self, x):
        return self.conv_block(x)


class MultiScaleModule(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.out_c = out_channel
        self.conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_5 = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )
        self.conv_7 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, self.out_c // 3, kernel_size=(2, 7)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_c // 3),
        )

    def forward(self, x):
        y1 = self.conv_3(x)
        y2 = self.conv_5(x)
        y3 = self.conv_7(x)
        return torch.cat([y1, y2, y3], dim=1)


class TinyMLP(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.mlp = nn.Sequential(
            nn.Linear(self.N, self.N // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.N // 4, self.N),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class AdaCorrModule(nn.Module):
    """Official ACM: FFT is performed on the last axis of each I/Q channel."""
    def __init__(self, N):
        super().__init__()
        self.Im = TinyMLP(N)
        self.Re = TinyMLP(N)

    def forward(self, x):
        # x: [B, 1, 2, L]. This is the official implementation's layout.
        x_init = copy.deepcopy(x)
        x_freq = torch.fft.fft(x, dim=-1)
        X_re = torch.real(x_freq)
        X_im = torch.imag(x_freq)
        h_re = self.Re(X_re)
        h_im = self.Im(X_im)
        corrected_freq = torch.mul(h_re, X_re) + 1j * torch.mul(h_im, X_im)
        corrected = torch.real(torch.fft.ifft(corrected_freq, dim=-1))
        return corrected + x_init


class FeaFusionModule(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(
            query_heads, key_heads.permute(0, 1, 3, 2)
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value_heads)
        shape = context.size()
        return context.contiguous().view(shape[0], -1, shape[-1])


class AMC_Net(nn.Module):
    """Official AMC-Net with an AMCLab-friendly constructor."""
    def __init__(
        self,
        num_classes=11,
        seq_length: int = 128,
        extend_channel=36,
        latent_dim=512,
        num_heads=2,
        conv_chan_list=None,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        # AMCLab unified constructor interface.
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected AMC_Net arguments: {unknown}")

        self.sig_len = int(seq_length)
        self.extend_channel = extend_channel
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.conv_chan_list = list(conv_chan_list or [36, 64, 128, 256])

        if self.conv_chan_list[0] != extend_channel:
            raise ValueError("conv_chan_list[0] must equal extend_channel")
        self.stem_layers_num = len(self.conv_chan_list) - 1

        self.ACM = AdaCorrModule(self.sig_len)
        self.MSM = MultiScaleModule(self.extend_channel)
        self.FFM = FeaFusionModule(
            self.num_heads, self.sig_len, self.sig_len
        )

        self.Conv_stem = nn.Sequential()
        for t in range(self.stem_layers_num):
            self.Conv_stem.add_module(
                f"conv_stem_{t}",
                Conv_Block(self.conv_chan_list[t], self.conv_chan_list[t + 1]),
            )

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.sig_len:
            raise ValueError(
                f"Expected input [B, 2, {self.sig_len}], got {tuple(x.shape)}"
            )

        # Preserve the official forward path exactly after adding channel dim.
        x = x.unsqueeze(1)                         # [B, 1, 2, L]
        x = self.ACM(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.MSM(x)
        x = self.Conv_stem(x)                      # [B, 256, 1, L]
        x = self.FFM(x.squeeze(2))                 # [B, 512, L/2]
        x = self.GAP(x)
        return self.classifier(x.squeeze(2))       # logits [B, C]


# AMCLab registry/import compatibility.
AMCNet = AMC_Net
