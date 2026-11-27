"""Built-in self-contained AMC baseline models (one model per file)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch.nn as nn

from .cgdnet import CGDNet
from .cldnn import CLDNN
from .cldnn2 import CLDNN2
from .cnn1 import CNN1
from .cnn2 import CNN2
from .dae import DAE
from .densenet import DenseNet
from .fea_t import FEAT, fea_t_base, fea_t_small
from .gru import GRU
from .ic_amcnet import ICAMCNet
from .iqformer import IQFormer, iqformer_base, iqformer_large, iqformer_small
from .lstm import LSTM
from .mcldnn import MCLDNN
from .mcnet import MCNet
from .pet_cgdnn import PETCGDNN
from .resnet import ResNet, resnet_base, resnet_large, resnet_small


# Canonical list of the 16 distinct baseline models (useful for full benchmark sweeps)
BENCHMARK_MODELS: List[str] = [
    "cnn1",
    "cnn2",
    "ic_amcnet",
    "mcnet",
    "resnet",
    "densenet",
    "gru",
    "lstm",
    "dae",
    "cldnn",
    "cldnn2",
    "mcldnn",
    "cgdnet",
    "pet_cgdnn",
    "iqformer",
    "fea_t",
]

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "cnn1": CNN1,
    "vtcnn2": CNN1,
    "cnn2": CNN2,
    "ic_amcnet": ICAMCNet,
    "mcnet": MCNet,
    "resnet": resnet_base,
    "resnet_small": resnet_small,
    "resnet_base": resnet_base,
    "resnet_large": resnet_large,
    "densenet": DenseNet,
    "gru": GRU,
    "lstm": LSTM,
    "dae": DAE,
    "cldnn": CLDNN,
    "cldnn2": CLDNN2,
    "mcldnn": MCLDNN,
    "cgdnet": CGDNet,
    "pet_cgdnn": PETCGDNN,
    "iqformer": iqformer_base,
    "iqformer_small": iqformer_small,
    "iqformer_base": iqformer_base,
    "iqformer_large": iqformer_large,
    "fea_t": fea_t_base,
    "fea_t_small": fea_t_small,
    "fea_t_base": fea_t_base,
}


def list_models(canonical_only: bool = False) -> List[str]:
    """Return registered baseline model names.

    Args:
        canonical_only: If True, returns the 16 canonical baseline names without size aliases.
    """
    if canonical_only:
        return list(BENCHMARK_MODELS)
    return sorted(MODEL_REGISTRY.keys())


def get_model(name: str, num_classes: int, seq_length: int = 128, **kwargs: Any) -> nn.Module:
    """Instantiate a registered baseline model by name."""
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_models()}")
    return MODEL_REGISTRY[key](num_classes=num_classes, seq_length=seq_length, **kwargs)


__all__ = [
    "CNN1", "CNN2", "ICAMCNet", "MCNet",
    "ResNet", "resnet_small", "resnet_base", "resnet_large",
    "DenseNet", "GRU", "LSTM", "DAE",
    "CLDNN", "CLDNN2", "MCLDNN", "CGDNet", "PETCGDNN",
    "IQFormer", "iqformer_small", "iqformer_base", "iqformer_large",
    "FEAT", "fea_t_small", "fea_t_base",
    "BENCHMARK_MODELS", "MODEL_REGISTRY", "list_models", "get_model",
]
