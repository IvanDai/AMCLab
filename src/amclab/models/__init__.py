"""Built-in standalone PyTorch AMC models."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Mapping,Optional
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
from .ea import EANet
from .mlea_amr import MLEAAMR

BENCHMARK_MODELS: List[str] = [
    "cnn1", "cnn2", 
    "ic_amcnet", "mcnet", "resnet", "densenet",
    "cgdnet", "1dcnn_pf",
    "gru", "lstm", "lstm_ap", 
    "cldnn", "cldnn2", "mcldnn", "pet_cgdnn", 
    "iqformer", "fea_t", "amcnet", "mcformer",
    "e_a","mlea_amr",
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
    "e_a":EANet,"mlea_amr":MLEAAMR,
}

def list_models(
    canonical_only: bool = False,
    registry: Optional[Mapping[str, Callable[..., nn.Module]]] = None,
) -> List[str]:
    active_registry = MODEL_REGISTRY if registry is None else registry
    if canonical_only:
        return list(BENCHMARK_MODELS)
    return sorted(active_registry.keys())


def get_model(
    name: str,
    num_classes: int,
    seq_length: int = 128,
    registry: Optional[Mapping[str, Callable[..., nn.Module]]] = None,
    **kwargs: Any,
) -> nn.Module:
    # Use built-in registry by default.
    active_registry = MODEL_REGISTRY if registry is None else registry
    key = name.strip().lower().replace("-", "_")
    if key not in active_registry:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {sorted(active_registry.keys())}"
        )
    model_cls = active_registry[key]
    return model_cls(
        num_classes=num_classes,
        seq_length=seq_length,
        **kwargs
    )


__all__ = [
    "CNN1", "CNN2", "ICAMCNet", "MCNet", "ResNet", "resnet_small",
    "resnet_base", "resnet_large", "DenseNet", "GRU", "LSTM", "LSTMAP",
    "CLDNN", "CLDNN2", "MCLDNN", "CGDNet", "PETCGDNN",
    "IQFormer", "iqformer_small", "iqformer_base", "iqformer_large",
    "FEAT", "fea_t_small", "fea_t_base", "AMCNet", "MCFormer", "DCNNPF",
    "EANet","MLEAAMR",
    "BENCHMARK_MODELS", "MODEL_REGISTRY", "list_models", "get_model",
]
