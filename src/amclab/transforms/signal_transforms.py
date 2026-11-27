"""Composable IQ signal transforms and training augmentations.

All transforms operate on a single IQ tensor of shape ``[2, L]`` (float32),
following a clean ``torchvision.transforms``-like API so DataLoaders stay pure IQ.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List

import torch


class Compose:
    """Chain multiple IQ transforms sequentially."""

    def __init__(self, transforms: Iterable[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms: List[Callable[[torch.Tensor], torch.Tensor]] = list(transforms)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            iq = t(iq)
        return iq

    def __repr__(self) -> str:
        body = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{body}])"


class UnitPowerNormalize:
    """Normalize each IQ frame to unit average complex power: E[I^2 + Q^2] = 1."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        power = (iq[0].square() + iq[1].square()).mean()
        return iq / torch.sqrt(power + self.eps)


class MaxAbsNormalize:
    """Scale each IQ frame so its maximum absolute value equals 1."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        peak = iq.abs().max()
        return iq / (peak + self.eps)


class ZeroMeanUnitVar:
    """Standardize I and Q channels independently to zero mean and unit variance."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        mean = iq.mean(dim=-1, keepdim=True)
        std = iq.std(dim=-1, keepdim=True, unbiased=False)
        return (iq - mean) / (std + self.eps)


class ToAmplitudePhase:
    """Convert raw IQ ``[2, L]`` into Amplitude/Phase ``[2, L]`` (phase normalized to [-1, 1])."""

    def __init__(self, normalize_phase: bool = True, eps: float = 1e-12):
        self.normalize_phase = bool(normalize_phase)
        self.eps = float(eps)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        i, q = iq[0], iq[1]
        amp = torch.sqrt(i.square() + q.square() + self.eps)
        phase = torch.atan2(q, i)
        if self.normalize_phase:
            phase = phase / math.pi
        return torch.stack((amp, phase), dim=0)


class RandomPhaseRotation:
    """Apply a random constant carrier phase rotation to an IQ sequence ``[2, L]``."""

    def __init__(self, p: float = 0.5, max_angle: float = math.pi):
        self.p = float(p)
        self.max_angle = float(max_angle)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return iq
        theta = (2.0 * torch.rand(1).item() - 1.0) * self.max_angle
        c, s = math.cos(theta), math.sin(theta)
        i, q = iq[0], iq[1]
        return torch.stack((i * c - q * s, i * s + q * c), dim=0)


class RandomTimeShift:
    """Apply a random circular time shift up to ``max_shift_ratio * L`` samples."""

    def __init__(self, p: float = 0.5, max_shift_ratio: float = 0.1):
        self.p = float(p)
        self.max_shift_ratio = float(max_shift_ratio)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return iq
        length = iq.shape[-1]
        max_shift = max(1, int(round(length * self.max_shift_ratio)))
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        return torch.roll(iq, shifts=shift, dims=-1)


class RandomIQImbalance:
    """Perturb I/Q gain balance by +/- ``max_db`` decibels."""

    def __init__(self, p: float = 0.5, max_db: float = 0.5):
        self.p = float(p)
        self.max_db = float(max_db)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return iq
        delta_db = (2.0 * torch.rand(1).item() - 1.0) * self.max_db
        gain = 10.0 ** (delta_db / 20.0)
        return torch.stack((iq[0] * gain, iq[1] / gain), dim=0)


class AdditiveGaussianNoise:
    """Add zero-mean isotropic Gaussian noise with standard deviation ``std``."""

    def __init__(self, p: float = 0.5, std: float = 0.05):
        self.p = float(p)
        self.std = float(std)

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return iq
        return iq + torch.randn_like(iq) * self.std
