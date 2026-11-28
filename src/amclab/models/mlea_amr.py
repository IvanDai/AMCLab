"""AMCLab port of the supplied paper-based MLEA-AMR reconstruction.
Default IQ-only average fusion; use_snr=True restores original SNR fusion.
Input [B,2,L], output logits [B,num_classes]. Not official author code.
"""
from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _make_1d_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    )


def _make_2d_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MBConv1D(nn.Module):
    """Expansion -> depthwise temporal convolution -> SE -> projection."""
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1,
        expand_ratio: float = 2.4, se_reduction: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        hidden = max(8, int(round(in_ch * expand_ratio)))
        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel, stride, kernel // 2,
                      groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.se = SqueezeExcite1D(hidden, reduction=se_reduction)
        self.project = nn.Sequential(
            nn.Conv1d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.dropout = nn.Dropout1d(dropout)
        self.use_residual = stride == 1 and in_ch == out_ch
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.se(self.depthwise(self.expand(x))))
        y = self.dropout(y)
        if self.use_residual:
            y = y + x
        return self.act(y)


class MBConv2D(nn.Module):
    """Expansion -> depthwise 2D convolution -> SE -> projection."""
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
        expand_ratio: float = 2.4, se_reduction: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        hidden = max(8, int(round(in_ch * expand_ratio)))
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.se = SqueezeExcite2D(hidden, reduction=se_reduction)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.use_residual = stride == 1 and in_ch == out_ch
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.se(self.depthwise(self.expand(x))))
        y = self.dropout(y)
        if self.use_residual:
            y = y + x
        return self.act(y)


class DSConv1D(nn.Module):
    """Depthwise-separable Conv1D, BN, GELU (paper MIDP encoder)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel, stride, padding=kernel // 2,
                      groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_iq(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Paper eqs. (4)-(5): per-segment unit average *complex* power."""
    if iq.ndim != 3:
        raise ValueError(f"I/Q must be 3D, got shape {tuple(iq.shape)}.")
    if iq.shape[1] != 2:
        if iq.shape[-1] == 2:
            iq = iq.transpose(1, 2)
        else:
            raise ValueError("I/Q must have shape [B,2,T] or [B,T,2].")
    if iq.shape[-1] < 8:
        raise ValueError("Signal length T must be >= 8.")
    power = (iq[:, 0].square() + iq[:, 1].square()).mean(dim=-1, keepdim=True)
    return iq / torch.sqrt(power.unsqueeze(1) + eps)


def construct_time_view(iq: torch.Tensor) -> torch.Tensor:
    """5 channels: I, Q, magnitude, delta-I, delta-Q; Fig. 3 / Sec. II-B."""
    i, q = iq[:, 0], iq[:, 1]
    amp = torch.sqrt(i.square() + q.square() + 1e-12)
    di = F.pad(i[:, 1:] - i[:, :-1], (1, 0))
    dq = F.pad(q[:, 1:] - q[:, :-1], (1, 0))
    return torch.stack((i, q, amp, di, dq), dim=1)


def construct_midp_view(iq: torch.Tensor, lags: Tuple[int, ...] = (1, 2, 4)
                        ) -> torch.Tensor:
    """
    9 channels with default lags: r, r^2, delta-r,
                               [cos(delta-theta_l), sin(delta-theta_l)] for l.
    At positions n < lag, phase-difference channels are zero padded.
    """
    i, q = iq[:, 0].float(), iq[:, 1].float()
    amp = torch.sqrt(i.square() + q.square() + 1e-12)
    phase = torch.atan2(q, i)
    dr = F.pad(amp[:, 1:] - amp[:, :-1], (1, 0))
    channels = [amp, amp.square(), dr]
    for lag in lags:
        if lag < 1 or lag >= amp.shape[-1]:
            raise ValueError(f"MIDP lag {lag} must lie in [1, T-1].")
        dphase = phase[:, lag:] - phase[:, :-lag]
        wrapped = torch.remainder(dphase + math.pi, 2 * math.pi) - math.pi
        channels.append(F.pad(torch.cos(wrapped), (lag, 0)))
        channels.append(F.pad(torch.sin(wrapped), (lag, 0)))
    return torch.stack(channels, dim=1).to(dtype=iq.dtype)


def construct_spectrum_view(
    iq: torch.Tensor, n_fft: int = 64, hop_length: int = 32,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    STFT on the COMPLEX normalized I/Q sequence. Two channels:
        log(|STFT| + eps), angle(STFT).
    torch.stft default center=True is an assumed implementation detail.
    """
    z = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    window = torch.hann_window(n_fft, device=iq.device, dtype=z.real.dtype)
    stft = torch.stft(z, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window, center=True,
                      return_complex=True)
    return torch.stack(
        (torch.log(stft.abs() + eps), torch.angle(stft)), dim=1
    ).to(dtype=iq.dtype)


class TemporalExpert(nn.Module):
    def __init__(self, width: int = 160, expand_ratio: float = 2.4,
                 se_reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        half = width // 2
        self.stem = _make_1d_bn_act(5, half, kernel=7, stride=2)
        self.block1 = MBConv1D(half, half, kernel=7, stride=1,
                               expand_ratio=expand_ratio,
                               se_reduction=se_reduction, dropout=dropout)
        self.block2 = MBConv1D(half, width, kernel=7, stride=2,
                               expand_ratio=expand_ratio,
                               se_reduction=se_reduction, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, time_view: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(self.stem(time_view)))
        return self.pool(x).flatten(1)


class SpectrumExpert(nn.Module):
    def __init__(self, width: int = 160, expand_ratio: float = 2.4,
                 se_reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        half = width // 2
        self.stem = _make_2d_bn_act(2, half, kernel=3, stride=2)
        self.block1 = MBConv2D(half, half, kernel=3, stride=1,
                               expand_ratio=expand_ratio,
                               se_reduction=se_reduction, dropout=dropout)
        self.block2 = MBConv2D(half, width, kernel=3, stride=2,
                               expand_ratio=expand_ratio,
                               se_reduction=se_reduction, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, spec_view: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(self.stem(spec_view)))
        return self.pool(x).flatten(1)


class ConstellationExpert(nn.Module):
    def __init__(self, width: int = 160, num_lags: int = 3):
        super().__init__()
        half = width // 2
        self.blocks = nn.Sequential(
            DSConv1D(3 + 2 * num_lags, half, kernel=5, stride=2),
            DSConv1D(half, width, kernel=5, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, midp_view: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(midp_view)).flatten(1)


class SNRConditionedFusion(nn.Module):
    """
    Paper Sec. III-C:
       a = softmax(cosine(q, f_i) / tau)
       y_attn = sum(a_i f_i)
       alpha = softmax(prior_MLP(f_snr))
       y_prior = sum(alpha_i f_i)
       f_fused = y_prior + gamma_ls * y_attn
    """
    def __init__(
        self, width: int = 160, snr_min: int = -20, snr_max: int = 18,
        snr_step: int = 2, prior_hidden: Optional[int] = None
    ):
        super().__init__()
        if (snr_max - snr_min) % snr_step:
            raise ValueError("SNR range must be a multiple of snr_step.")
        self.snr_min, self.snr_max, self.snr_step = snr_min, snr_max, snr_step
        n_bins = (snr_max - snr_min) // snr_step + 1
        self.embedding = nn.Embedding(n_bins, width)
        self.snr_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        ph = prior_hidden or (width // 2)
        self.prior_mlp = nn.Sequential(
            nn.Linear(width, ph),
            nn.GELU(),
            nn.Linear(ph, 3),
        )
        self.query_norm = nn.LayerNorm(width)
        self.expert_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        # Learnable positive temperature via log-scale parameterization.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0)))
        self.layer_scale = nn.Parameter(torch.full((width,), 0.01))

    def snr_to_index(self, snr_db: torch.Tensor) -> torch.Tensor:
        return torch.round(
            (snr_db.float() - self.snr_min) / self.snr_step
        ).long().clamp_(0, self.embedding.num_embeddings - 1)

    def forward(self, expert_features: torch.Tensor, snr_db: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if expert_features.ndim != 3 or expert_features.shape[1] != 3:
            raise ValueError("Expert features must have shape [B,3,width].")
        batch_size = expert_features.shape[0]
        snr_db = torch.as_tensor(snr_db, device=expert_features.device)
        if snr_db.numel() != batch_size:
            raise ValueError("Please supply exactly one SNR value per sample.")
        snr_db = snr_db.reshape(batch_size)
        f_snr = self.snr_mlp(self.embedding(self.snr_to_index(snr_db)))

        q = F.normalize(self.query_norm(f_snr), dim=-1)
        keys = torch.stack(
            [F.normalize(norm(expert_features[:, idx]), dim=-1)
             for idx, norm in enumerate(self.expert_norms)], dim=1
        )
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        scores = (keys * q.unsqueeze(1)).sum(dim=-1) / temperature
        attn_weights = torch.softmax(scores, dim=1)
        y_attn = torch.einsum("bk,bkc->bc", attn_weights, expert_features)

        prior_weights = torch.softmax(self.prior_mlp(f_snr), dim=1)
        y_prior = torch.einsum("bk,bkc->bc", prior_weights, expert_features)
        fused = y_prior + self.layer_scale.unsqueeze(0) * y_attn

        return fused, {"attention_weights": attn_weights,
                       "prior_weights": prior_weights,
                       "snr_embedding": f_snr}


class MLEAAMR(nn.Module):
    """MLEA-AMR paper-based reconstruction (not a pretrained author checkpoint)."""
    def __init__(
        self, num_classes: int = 11, seq_length: int = 128, *,
        use_snr: bool = False, width: int = 160,
        expand_ratio: float = 2.4, se_reduction: int = 4,
        block_dropout: float = 0.1, head_dropout: float = 0.2,
        n_fft: int = 64, hop_length: int = 32, midp_lags: Tuple[int, ...] = (1, 2, 4),
        snr_min: int = -20, snr_max: int = 18, snr_step: int = 2,
        prior_hidden: Optional[int] = None
    ):
        super().__init__()
        if width < 16 or width % 2:
            raise ValueError("width must be an even integer >= 16.")
        self.seq_length = int(seq_length)
        self.use_snr = bool(use_snr)
        if n_fft < 2 or hop_length < 1 or self.seq_length <= n_fft // 2:
            raise ValueError("seq_length must exceed n_fft // 2 for centered STFT.")
        if self.seq_length < 8 or any(lag < 1 or lag >= self.seq_length for lag in midp_lags):
            raise ValueError("Invalid sequence length or MIDP lags.")
        if self.use_snr and (snr_step <= 0 or snr_max < snr_min):
            raise ValueError("Invalid SNR bin configuration.")
        self.n_fft, self.hop_length = n_fft, hop_length
        self.midp_lags = tuple(midp_lags)
        self.time_expert = TemporalExpert(width, expand_ratio, se_reduction, block_dropout)
        self.spectrum_expert = SpectrumExpert(width, expand_ratio, se_reduction, block_dropout)
        self.constellation_expert = ConstellationExpert(width, len(midp_lags))
        self.fusion = SNRConditionedFusion(
            width, snr_min=snr_min, snr_max=snr_max,
            snr_step=snr_step, prior_hidden=prior_hidden
        ) if self.use_snr else None
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(head_dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(
        self, iq: torch.Tensor, snr_db: Optional[torch.Tensor] = None,
        return_details: bool = False
    ):
        if iq.ndim != 3 or iq.shape[1:] != (2, self.seq_length):
            raise ValueError(f"Expected IQ [B,2,{self.seq_length}], got {tuple(iq.shape)}")
        if self.use_snr and snr_db is None:
            raise ValueError("use_snr=True requires one SNR value in dB per sample.")
        if self.use_snr and not torch.isfinite(torch.as_tensor(snr_db)).all():
            raise ValueError("SNR values must be finite.")
        iq = normalize_iq(iq)
        f_time = self.time_expert(construct_time_view(iq))
        f_spec = self.spectrum_expert(
            construct_spectrum_view(iq, self.n_fft, self.hop_length)
        )
        f_const = self.constellation_expert(
            construct_midp_view(iq, self.midp_lags)
        )
        stacked = torch.stack((f_time, f_spec, f_const), dim=1)
        if self.use_snr:
            fused, info = self.fusion(stacked, snr_db)
        else:
            fused = stacked.mean(dim=1)
            info = {"fusion_weights": stacked.new_full((stacked.shape[0], 3), 1 / 3)}
        logits = self.classifier(fused)
        if return_details:
            return logits, {**info, "expert_features": stacked, "fused_features": fused}
        return logits


def signal_aware_augmentation(
    iq: torch.Tensor, probability: float = 0.2,
    max_shift_ratio: float = 0.08, max_iq_imbalance_db: float = 0.5,
    max_phase_rad: float = math.pi / 8,
) -> torch.Tensor:
    """
    Paper Sec. III-A / IV-A: OPTIONAL, training-only signal perturbations.
    With probability 0.2 per sample choose ONE of:
      circular time shift, I/Q gain imbalance, constant phase rotation.
    This function is vectorized and preserves [B,2,T] or [B,T,2].
    Call BEFORE model(...) in your training loop; never on test data.
    """
    if iq.ndim != 3:
        raise ValueError("iq must be a 3D tensor")
    transpose_back = iq.shape[1] != 2 and iq.shape[-1] == 2
    x = iq.transpose(1, 2) if transpose_back else iq
    if x.shape[1] != 2:
        raise ValueError("Expected [B,2,T] or [B,T,2]")
    batch, _, length = x.shape
    device = x.device
    apply = torch.rand(batch, device=device) < probability
    modes = torch.randint(0, 3, (batch,), device=device)

    # Per-example circular time shift, at most 8% of the input length.
    max_shift = max(1, round(length * max_shift_ratio))
    shifts = torch.randint(-max_shift, max_shift + 1, (batch,), device=device)
    pos = (torch.arange(length, device=device)[None] - shifts[:, None]) % length
    shifted = x.gather(2, pos[:, None].expand(-1, 2, -1))

    # I*gain, Q/gain; gain range corresponds to +/-0.5 dB.
    delta_db = (2 * torch.rand(batch, device=device) - 1) * max_iq_imbalance_db
    gain = torch.pow(10.0, delta_db / 20.0).to(x.dtype)
    scaled = x * torch.stack((gain, gain.reciprocal()), dim=1).unsqueeze(-1)

    # Global complex-plane phase rotation.
    phase = (2 * torch.rand(batch, device=device) - 1) * max_phase_rad
    c, ss = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    rotated = torch.stack((c * x[:, 0] - ss * x[:, 1],
                           ss * x[:, 0] + c * x[:, 1]), dim=1)

    out = x
    out = torch.where((apply & (modes == 0))[:, None, None], shifted, out)
    out = torch.where((apply & (modes == 1))[:, None, None], scaled, out)
    out = torch.where((apply & (modes == 2))[:, None, None], rotated, out)
    return out.transpose(1, 2) if transpose_back else out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MLEA_AMR = MLEAAMR

if __name__ == "__main__":
    for use_snr in (False, True):
        model = MLEAAMR(use_snr=use_snr)
        iq = torch.randn(4, 2, 128)
        snr = torch.tensor([-20., -6., 8., 18.]) if use_snr else None
        logits = model(iq, snr)
        logits.square().mean().backward()
        print(use_snr, logits.shape, count_parameters(model))
