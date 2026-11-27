"""Self-contained Frame-wise Embedding Aided Transformer (FEA-T, Chen et al.).

Automatically scales ``patch_size`` for long sequences (e.g., L=1024 in RML2018)
and exposes small/base variants in the same file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class _MHSABlock(nn.Module):
    def __init__(self, d_model: int, d_head: int = 16, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.scale = d_head ** 0.5
        self.to_q = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.to_k = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.to_v = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.proj = nn.Linear(d_head * num_heads, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, p, _ = x.shape
        q = self.to_q(x).reshape(b, p, self.num_heads, self.d_head).transpose(1, 2) / self.scale
        k = self.to_k(x).reshape(b, p, self.num_heads, self.d_head).transpose(1, 2)
        v = self.to_v(x).reshape(b, p, self.num_heads, self.d_head).transpose(1, 2)
        attn = self.drop(torch.softmax(q @ k.transpose(-2, -1), dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, p, self.num_heads * self.d_head)
        return self.drop(self.proj(out))


class _DBGLUBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.half = d_hidden // 2
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.half, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h1, h2 = h[..., : self.half], h[..., self.half :]
        return self.fc2(self.drop(h1 * self.act(h2) + h2 * self.act(h1)))


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_head: int, d_hidden: int, num_heads: int, dropout: float):
        super().__init__()
        self.mhsa = _MHSABlock(d_model, d_head=d_head, num_heads=num_heads, dropout=dropout)
        self.ffn = _DBGLUBlock(d_model, d_hidden=d_hidden, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.mhsa(x))
        return self.norm2(x + self.ffn(x))


class FEAT(nn.Module):
    """Frame-wise Embedding Aided Transformer (FEA-T) for IQ signals ``[B, 2, L]``."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        patch_size: int | None = None,
        overlap: float = 0.5,
        layer_num: int = 4,
        num_heads: int = 4,
        d_head: int = 16,
        hidden_features: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if patch_size is None:
            patch_size = 64 if seq_length >= 512 else 16
        self.patch_size = int(patch_size)
        self.stride = max(1, int(self.patch_size * overlap))
        self.frame_num = max(1, (seq_length - self.patch_size) // self.stride + 1)
        d_model = self.patch_size * 2

        self.embed = nn.Linear(d_model, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.frame_num + 1, d_model))
        self.layers = nn.ModuleList(
            [_TransformerLayer(d_model, d_head, hidden_features, num_heads, dropout) for _ in range(layer_num)]
        )
        self.classifier = nn.Linear(d_model, num_classes)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L] -> unfold sliding frames
        b, _, length = x.shape
        frames = x.unfold(dimension=2, size=self.patch_size, step=self.stride)  # [B, 2, F, P]
        frames = frames.permute(0, 2, 1, 3).reshape(b, -1, 2 * self.patch_size)
        tokens = self.embed(frames)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((tokens, cls), dim=1)
        # Gracefully interpolate positional embeddings if runtime sequence length differs
        if tokens.shape[1] == self.pos_embed.shape[1]:
            tokens = tokens + self.pos_embed
        else:
            pe = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=tokens.shape[1], mode="linear", align_corners=False
            ).transpose(1, 2)
            tokens = tokens + pe
        for layer in self.layers:
            tokens = layer(tokens)
        return self.classifier(tokens[:, -1])


def fea_t_small(num_classes: int = 11, seq_length: int = 128, **kwargs) -> FEAT:
    return FEAT(num_classes=num_classes, seq_length=seq_length, layer_num=4, hidden_features=128, **kwargs)


def fea_t_base(num_classes: int = 11, seq_length: int = 128, **kwargs) -> FEAT:
    return FEAT(num_classes=num_classes, seq_length=seq_length, layer_num=8, hidden_features=256, **kwargs)
