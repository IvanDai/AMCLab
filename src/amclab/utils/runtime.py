"""Hardware device auto-detection (CUDA / Apple MPS / CPU) and reproducibility utilities."""

from __future__ import annotations

import os
import platform
import random
from typing import Any, Dict

import numpy as np
import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve computation device automatically or validate an explicit request.

    Priority when ``device="auto"``:
        1. NVIDIA CUDA (``cuda``)
        2. Apple Silicon Metal Performance Shaders (``mps``)
        3. CPU (``cpu``)
    """
    if isinstance(device, torch.device):
        return device

    req = (device or "auto").strip().lower()
    if req != "auto":
        dev = torch.device(req)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
        if dev.type == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("Requested device 'mps', but Apple MPS is not available.")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """Return a concise human-readable hardware description."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        return f"CUDA ({props.name}, {vram_gb:.1f} GB VRAM)"
    if device.type == "mps":
        chip = platform.processor() or platform.machine()
        return f"Apple MPS ({chip})"
    return f"CPU ({platform.processor() or platform.machine()})"


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_state() -> Dict[str, Any]:
    """Capture RNG states across Python, NumPy, and PyTorch for exact resume."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any] | None) -> None:
    """Restore RNG states captured by :func:`get_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count total (or trainable) parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def doctor() -> Dict[str, Any]:
    """Inspect current environment, hardware acceleration, and optional packages."""
    dev = resolve_device("auto")
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "auto_device": str(dev),
        "device_description": describe_device(dev),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    for pkg in ("numpy", "scipy", "h5py", "tqdm", "matplotlib"):
        try:
            __import__(pkg)
            info[f"pkg_{pkg}"] = "installed"
        except ImportError:
            info[f"pkg_{pkg}"] = "missing"
    return info
