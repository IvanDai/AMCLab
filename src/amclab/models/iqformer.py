"""Self-contained IQFormer (Shao et al., IEEE TCCN 2024).

Computes its STFT time-frequency branch internally from raw IQ ``[B, 2, L]``
so its public signature strictly follows ``forward(x: Tensor) -> Tensor``.
Includes lightweight, base, and large variants in this single module.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
        return x.div(keep) * mask


class _PatchEmbed1D(nn.Module):
    def __init__(self, patch_size: int, stride: int, padding: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class _ConvEncoderIQ(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, kernel_size: int = 3, drop_path: float = 0.0, use_layer_scale: bool = True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        if self.use_layer_scale:
            y = self.layer_scale * y
        return x + self.drop_path(y)


class _FCN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EfficientAdditiveAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        self.to_query = nn.Linear(dim, dim * num_heads)
        self.to_key = nn.Linear(dim, dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(dim * num_heads, 1))
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim * num_heads, dim * num_heads)
        self.final = nn.Linear(dim * num_heads, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        q = torch.nn.functional.normalize(self.to_query(x), dim=-1)
        k = torch.nn.functional.normalize(self.to_key(x), dim=-1)
        a = torch.nn.functional.normalize((q @ self.w_g) * self.scale, dim=1)
        g = torch.sum(a * q, dim=1, keepdim=True).expand(-1, k.shape[1], -1)
        return self.final(self.proj(g * k) + q)


class _IQFormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.local_rep = _ConvEncoderIQ(dim, dim, kernel_size=3, drop_path=0.0, use_layer_scale=True)
        self.attn = _EfficientAdditiveAttention(dim, num_heads=1)
        self.fcn = _FCN(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim, 1))
            self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_rep(x)
        attn_out = self.attn(x.transpose(1, 2)).transpose(1, 2)
        if self.use_layer_scale:
            x = x + self.drop_path(self.ls1 * attn_out)
            x = x + self.drop_path(self.ls2 * self.fcn(x))
        else:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.fcn(x))
        return x


class IQFormer(nn.Module):
    """Multi-modality IQ + STFT Transformer for Automatic Modulation Classification."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        layers: Sequence[int] = (2, 2, 2),
        embed_dims: Sequence[int] = (64, 64, 64),
        mlp_ratio: float = 4.0,
        down_patch_size: int = 5,
        down_stride: int = 3,
        down_pad: int = 1,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        stft_freq_bins: int = 32,
        vit_num: int = 1,
    ):
        super().__init__()
        self.stft_freq_bins = int(stft_freq_bins)
        self.bn_iq = nn.BatchNorm1d(2)
        self.bn_stft = nn.BatchNorm2d(1)

        half_dim = embed_dims[0] // 8
        self.stem_iq = nn.Sequential(
            nn.Conv1d(2, half_dim, kernel_size=5, stride=1, padding=2, groups=2),
            nn.BatchNorm1d(half_dim),
        )
        self.stem_stft = nn.Sequential(
            nn.Conv2d(1, half_dim, kernel_size=(self.stft_freq_bins, 1), stride=1),
            nn.BatchNorm2d(half_dim),
            nn.ReLU(inplace=True),
        )
        fused_in = half_dim * 2
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_in, embed_dims[0] // 2, kernel_size=1),
            nn.BatchNorm1d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv1d(embed_dims[0] // 2, embed_dims[0] // 2, kernel_size=1),
            nn.Dropout(drop_rate),
        )
        self.patch_lstm = nn.LSTM(
            input_size=embed_dims[0] // 2,
            hidden_size=embed_dims[0] // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=drop_rate,
        )

        # Build hierarchical Conv + IQFormer stages
        total_blocks = max(sum(layers) - 1, 1)
        stages = []
        block_cursor = 0
        for stage_idx, depth in enumerate(layers):
            dim = embed_dims[stage_idx]
            blocks = []
            for b_idx in range(depth):
                dpr = drop_path_rate * (block_cursor + b_idx) / total_blocks
                if depth - b_idx <= vit_num:
                    blocks.append(_IQFormerBlock(dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr))
                else:
                    blocks.append(_ConvEncoderIQ(dim, int(dim * mlp_ratio), kernel_size=3))
            block_cursor += depth
            stages.append(nn.Sequential(*blocks))
            if stage_idx < len(layers) - 1 and embed_dims[stage_idx] != embed_dims[stage_idx + 1]:
                stages.append(
                    _PatchEmbed1D(down_patch_size, down_stride, down_pad, embed_dims[stage_idx], embed_dims[stage_idx + 1])
                )
        self.stages = nn.ModuleList(stages)
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _compute_stft(self, iq: torch.Tensor) -> torch.Tensor:
        length = iq.shape[-1]
        n_fft = (self.stft_freq_bins - 1) * 2
        window = torch.hann_window(n_fft, device=iq.device, dtype=torch.float32)
        z = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        spec = torch.stft(
            z, n_fft=n_fft, hop_length=1, win_length=n_fft,
            window=window, center=True, return_complex=True,
        )
        return spec.abs()[:, : self.stft_freq_bins, :length].unsqueeze(1).to(dtype=iq.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft = self._compute_stft(x)
        f_iq = self.stem_iq(self.bn_iq(x))
        f_stft = self.stem_stft(self.bn_stft(stft)).squeeze(2)
        h = self.fusion(torch.cat((f_iq, f_stft), dim=1))
        h, _ = self.patch_lstm(h.transpose(1, 2))
        h = h.transpose(1, 2)
        for stage in self.stages:
            h = stage(h)
        return self.head(self.pool(self.norm(h)))


def iqformer_small(num_classes: int = 11, seq_length: int = 128, **kwargs) -> IQFormer:
    return IQFormer(num_classes=num_classes, seq_length=seq_length, layers=(1, 1, 1), embed_dims=(32, 32, 32), **kwargs)


def iqformer_base(num_classes: int = 11, seq_length: int = 128, **kwargs) -> IQFormer:
    return IQFormer(num_classes=num_classes, seq_length=seq_length, layers=(2, 3, 2), embed_dims=(64, 64, 64), **kwargs)


def iqformer_large(num_classes: int = 11, seq_length: int = 128, **kwargs) -> IQFormer:
    return IQFormer(num_classes=num_classes, seq_length=seq_length, layers=(2, 3, 2), embed_dims=(64, 96, 128), **kwargs)
