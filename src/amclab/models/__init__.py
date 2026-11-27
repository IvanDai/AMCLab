"""Built-in standalone PyTorch AMC models."""
from __future__ import annotations
from typing import Any, Callable, Dict, List
import torch.nn as nn

from .amcnet import AMCNet
from .cgdnet import CGDNet
from .cldnn import CLDNN
from .cldnn2 import CLDNN2
from .cnn1 import CNN1
from .cnn2 import CNN2
from .dcnn_pf import DCNNPF
from .densenet import DenseNet
from .fea_t import FEAT, fea_t_base, fea_t_small
from .gru import GRU
from .ic_amcnet import ICAMCNet
from .iqformer import IQFormer, iqformer_base, iqformer_large, iqformer_small
from .lstm import LSTM, LSTMAP
from .mcformer import MCFormer
from .mcldnn import MCLDNN
from .mcnet import MCNet
from .pet_cgdnn import PETCGDNN
from .resnet import ResNet, resnet_base, resnet_large, resnet_small

BENCHMARK_MODELS: List[str] = [
    "cnn1", "cnn2", "ic_amcnet", "mcnet", "resnet", "densenet",
    "gru", "lstm", "lstm_ap", "cldnn", "cldnn2", "mcldnn", "cgdnet",
    "pet_cgdnn", "iqformer", "fea_t", "amcnet", "mcformer", "1dcnn_pf",
]

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "cnn1": CNN1, "vtcnn2": CNN1,
    "cnn2": CNN2, "ic_amcnet": ICAMCNet, "mcnet": MCNet,
    "resnet": ResNet, "resnet_small": resnet_small,
    "resnet_base": resnet_base, "resnet_large": resnet_large,
    "densenet": DenseNet, "gru": GRU, "lstm": LSTM, "lstm_ap": LSTMAP,
    "cldnn": CLDNN, "cldnn2": CLDNN2, "mcldnn": MCLDNN,
    "cgdnet": CGDNet, "pet_cgdnn": PETCGDNN,
    "iqformer": iqformer_base, "iqformer_small": iqformer_small,
    "iqformer_base": iqformer_base, "iqformer_large": iqformer_large,
    "fea_t": fea_t_base, "fea_t_small": fea_t_small, "fea_t_base": fea_t_base,
    "amcnet": AMCNet, "mcformer": MCFormer,
    "1dcnn_pf": DCNNPF, "dcnn_pf": DCNNPF,
}

def list_models(canonical_only: bool = False) -> List[str]:
    return list(BENCHMARK_MODELS) if canonical_only else sorted(MODEL_REGISTRY)

def get_model(name: str, num_classes: int, seq_length: int = 128, **kwargs: Any) -> nn.Module:
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {list_models()}")
    return MODEL_REGISTRY[key](num_classes=num_classes, seq_length=seq_length, **kwargs)

__all__ = [
    "CNN1", "CNN2", "ICAMCNet", "MCNet", "ResNet", "resnet_small",
    "resnet_base", "resnet_large", "DenseNet", "GRU", "LSTM", "LSTMAP",
    "CLDNN", "CLDNN2", "MCLDNN", "CGDNet", "PETCGDNN",
    "IQFormer", "iqformer_small", "iqformer_base", "iqformer_large",
    "FEAT", "fea_t_small", "fea_t_base", "AMCNet", "MCFormer", "DCNNPF",
    "BENCHMARK_MODELS", "MODEL_REGISTRY", "list_models", "get_model",
]
