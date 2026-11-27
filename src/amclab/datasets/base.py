"""Base AMC dataset abstractions, pure-IQ sample contract, and stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetMeta:
    """Immutable metadata describing a loaded AMC dataset."""

    name: str
    num_samples: int
    num_classes: int
    class_names: Tuple[str, ...]
    seq_length: int
    snr_values: Tuple[float, ...]
    storage_backend: str
    file_paths: Tuple[str, ...]


def normalize_iq_layout(raw: np.ndarray, layout: str = "auto") -> np.ndarray:
    """Ensure a single IQ sample has contiguous float32 shape ``(2, L)``."""
    arr = np.asarray(raw)
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        if arr.ndim != 1:
            raise ValueError(f"Complex IQ sample must be 1D [L], got {arr.shape}.")
        arr = np.stack((arr.real, arr.imag), axis=0)
    if arr.ndim != 2:
        raise ValueError(f"IQ sample must be 2D [2, L] or [L, 2], got {arr.shape}.")

    if layout == "channels_last":
        if arr.shape[1] != 2:
            raise ValueError(f"Expected channels_last [L, 2], got {arr.shape}.")
        arr = arr.T
    elif layout == "channels_first":
        if arr.shape[0] != 2:
            raise ValueError(f"Expected channels_first [2, L], got {arr.shape}.")
    elif layout == "auto":
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            pass
        elif arr.shape[1] == 2 and arr.shape[0] != 2:
            arr = arr.T
        else:
            raise ValueError(f"Ambiguous IQ shape {arr.shape}; please specify layout explicitly.")
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return np.ascontiguousarray(arr, dtype=np.float32)


def labels_to_1d(raw: Any) -> np.ndarray:
    """Convert one-hot ``[N, C]`` or 2D column ``[N, 1]`` labels to 1D int64 indices."""
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1).astype(np.int64)
    if arr.ndim in (1, 2):
        return arr.reshape(-1)
    raise ValueError(f"Unsupported label array shape: {arr.shape}")


def encode_labels(
    raw: Any,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Map arbitrary integer or string labels to contiguous ``0 .. C-1`` indices."""
    raw_1d = labels_to_1d(raw)
    values, inv = np.unique(raw_1d, return_inverse=True)
    if class_names is None:
        names = tuple(str(v) for v in values)
        return inv.astype(np.int64), names

    names_list = [str(c) for c in class_names]
    if len(names_list) != len(values):
        if len(names_list) > len(values) and np.issubdtype(values.dtype, np.integer):
            names_list = [names_list[int(v)] if int(v) < len(names_list) else str(v) for v in values]
        else:
            raise ValueError(
                f"Provided class_names length ({len(names_list)}) does not match unique labels ({len(values)})."
            )

    if np.asarray(values).dtype.kind in ("U", "S", "O") and set(str(v) for v in values) == set(names_list):
        lookup = {name: i for i, name in enumerate(names_list)}
        inv = np.array([lookup[str(v)] for v in raw_1d], dtype=np.int64)

    return inv.astype(np.int64), tuple(names_list)


class BaseAMCDataset(Dataset):
    """Base PyTorch Dataset emitting pure IQ ``[2, L]``, integer label, and scalar SNR.

    Every ``__getitem__`` returns a tuple ``(iq, label, snr)`` where:
        - ``iq``: ``torch.FloatTensor`` of shape ``[2, L]`` (transformed if ``transform`` is set)
        - ``label``: ``int`` class index in ``[0, num_classes - 1]``
        - ``snr``: ``float`` signal-to-noise ratio in dB (or ``0.0`` if unavailable)
    """

    def __init__(self, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.transform = transform
        self.labels: np.ndarray = np.empty(0, dtype=np.int64)
        self.snrs: np.ndarray = np.empty(0, dtype=np.float32)
        self.class_names: Tuple[str, ...] = ()
        self.predefined_test_indices: Optional[np.ndarray] = None
        self.meta: Optional[DatasetMeta] = None

    def _init_metadata(
        self,
        name: str,
        file_paths: Sequence[str | Path],
        raw_labels: Any,
        raw_snrs: Any,
        class_names: Optional[Sequence[str]] = None,
        storage_backend: str = "lazy",
    ) -> None:
        self.labels, self.class_names = encode_labels(raw_labels, class_names)
        if raw_snrs is None:
            self.snrs = np.zeros(len(self.labels), dtype=np.float32)
        else:
            self.snrs = np.asarray(raw_snrs, dtype=np.float32).reshape(-1)
            if len(self.snrs) != len(self.labels):
                raise ValueError(f"SNR count ({len(self.snrs)}) != label count ({len(self.labels)}).")
        if len(self.labels) == 0:
            raise ValueError(f"Dataset '{name}' contains zero samples.")

        sample0 = self._read_iq(0)
        seq_length = int(sample0.shape[-1])
        finite_snrs = self.snrs[np.isfinite(self.snrs)]
        snr_values = tuple(sorted(float(s) for s in np.unique(finite_snrs))) if len(finite_snrs) > 0 else ()

        self.meta = DatasetMeta(
            name=name,
            num_samples=len(self.labels),
            num_classes=len(self.class_names),
            class_names=self.class_names,
            seq_length=seq_length,
            snr_values=snr_values,
            storage_backend=storage_backend,
            file_paths=tuple(str(Path(p).resolve()) for p in file_paths),
        )

    @property
    def num_classes(self) -> int:
        assert self.meta is not None
        return self.meta.num_classes

    @property
    def seq_length(self) -> int:
        assert self.meta is not None
        return self.meta.seq_length

    @property
    def snr_values(self) -> Tuple[float, ...]:
        assert self.meta is not None
        return self.meta.snr_values

    def __len__(self) -> int:
        return len(self.labels)

    def _read_iq(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        idx = int(idx)
        iq = torch.from_numpy(self._read_iq(idx))
        if self.transform is not None:
            iq = self.transform(iq)
        return iq, int(self.labels[idx]), float(self.snrs[idx])


class AMCSubset(Dataset):
    """Lightweight view over a ``BaseAMCDataset`` with an independent ``transform`` hook.

    Allows applying data augmentation ONLY to the training split while keeping
    validation and test splits deterministic and unaugmented.
    """

    def __init__(
        self,
        dataset: BaseAMCDataset,
        indices: Sequence[int] | np.ndarray,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.dataset.class_names

    @property
    def seq_length(self) -> int:
        return self.dataset.seq_length

    @property
    def snr_values(self) -> Tuple[float, ...]:
        return self.dataset.snr_values

    @property
    def meta(self) -> Optional[DatasetMeta]:
        return self.dataset.meta

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        real_idx = int(self.indices[idx])
        iq = torch.from_numpy(self.dataset._read_iq(real_idx))
        if self.transform is not None:
            iq = self.transform(iq)
        elif self.dataset.transform is not None:
            iq = self.dataset.transform(iq)
        return iq, int(self.dataset.labels[real_idx]), float(self.dataset.snrs[real_idx])


def stratified_split_indices(
    labels: np.ndarray,
    snrs: np.ndarray,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    predefined_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically partition indices stratified jointly by ``(modulation, SNR)``."""
    r_train, r_val, r_test = split
    total = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / total, r_val / total, r_test / total

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(labels), dtype=np.int64)

    if predefined_test is not None and len(predefined_test) > 0:
        test_arr = np.asarray(predefined_test, dtype=np.int64)
        pool = np.setdiff1d(all_idx, test_arr, assume_unique=True)
        train_frac = r_train / max(r_train + r_val, 1e-8)
        groups: Dict[Tuple[int, float], List[int]] = {}
        for i in pool:
            groups.setdefault((int(labels[i]), float(snrs[i])), []).append(int(i))
        tr_parts, va_parts = [], []
        for grp in groups.values():
            arr = np.asarray(grp, dtype=np.int64)
            rng.shuffle(arr)
            n = len(arr)
            cut = max(1, min(n - 1, int(round(n * train_frac)))) if n >= 2 else n
            tr_parts.append(arr[:cut])
            va_parts.append(arr[cut:])
        train_arr = np.concatenate(tr_parts)
        val_arr = np.concatenate(va_parts)
    else:
        groups = {}
        for i in all_idx:
            groups.setdefault((int(labels[i]), float(snrs[i])), []).append(int(i))
        tr_parts, va_parts, te_parts = [], [], []
        for grp in groups.values():
            arr = np.asarray(grp, dtype=np.int64)
            rng.shuffle(arr)
            n = len(arr)
            if n < 3:
                raise ValueError(f"Stratified cell has only {n} samples (<3); cannot split into train/val/test.")
            n_tr = max(1, int(round(n * r_train)))
            n_va = max(1, int(round(n * r_val)))
            if n_tr + n_va >= n:
                n_tr = max(1, n - 2)
                n_va = 1
            tr_parts.append(arr[:n_tr])
            va_parts.append(arr[n_tr : n_tr + n_va])
            te_parts.append(arr[n_tr + n_va :])
        train_arr = np.concatenate(tr_parts)
        val_arr = np.concatenate(va_parts)
        test_arr = np.concatenate(te_parts)

    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    rng.shuffle(test_arr)
    return train_arr, val_arr, test_arr


def save_split_indices(
    path: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Path:
    """Persist train/val/test index arrays to a compressed ``.npz`` file."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
    )
    return p


def load_split_indices(
    path: str | Path,
    total_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate disjoint train/val/test indices from a ``.npz`` split file."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Split file not found: {p}")
    with np.load(p, allow_pickle=False) as obj:
        train_idx = np.asarray(obj["train"], dtype=np.int64)
        val_idx = np.asarray(obj["val"], dtype=np.int64)
        test_idx = np.asarray(obj["test"], dtype=np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Split file {p} contains an empty partition.")
    combined = np.concatenate([train_idx, val_idx, test_idx])
    if (
        len(combined) != total_samples
        or combined.min() < 0
        or combined.max() >= total_samples
        or len(np.unique(combined)) != total_samples
    ):
        raise ValueError(
            f"Split file {p} does not match dataset size ({total_samples}) or has overlapping indices."
        )
    return train_idx, val_idx, test_idx
