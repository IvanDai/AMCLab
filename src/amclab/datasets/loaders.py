"""Concrete lazy / out-of-core dataset loaders for standard AMC benchmarks.

Supported datasets:
    - ``rml2016.10a``  : RadioML 2016.10a (11 classes, L=128, Pickle -> disk memmap cache)
    - ``rml2016.10b``  : RadioML 2016.10b (10 classes, L=128, Pickle -> disk memmap cache)
    - ``rml22``        : RadioML 22.01A   (10/11 classes, L=128, Pickle -> disk memmap cache)
    - ``rml2018``      : RadioML 2018.01a (24 classes, L=1024, True Lazy HDF5)
    - ``hisarmod2019`` : HisarMod 2019.1  (26 classes, L=1024, Lazy HDF5 or MAT -> memmap)
    - ``synthetic``    : Built-in synthetic IQ dataset (any length/classes, zero external files)
"""

from __future__ import annotations

import bisect
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import (
    AMCSubset,
    BaseAMCDataset,
    labels_to_1d,
    load_split_indices,
    normalize_iq_layout,
    save_split_indices,
    stratified_split_indices,
)


RML2018_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

HISARMOD_CLASSES = [
    "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "64PSK",
    "4QAM", "8QAM", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "2FSK", "4FSK", "8FSK", "16FSK",
    "4PAM", "8PAM", "16PAM",
    "AM-DSB", "AM-DSB-SC", "AM-USB", "AM-LSB", "FM", "PM",
]


def _memmap_cache_dir(source_path: Path, layout: str, cache_root: Optional[str | Path] = None) -> Path:
    st = source_path.stat()
    sig = f"{source_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{layout}"
    digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
    root = Path(cache_root).expanduser().resolve() if cache_root else (source_path.parent / ".amclab_cache")
    return root / f"{source_path.stem}_{digest}"


class RMLPickleDataset(BaseAMCDataset):
    """Loader for RadioML 2016.10a, 2016.10b, and RML22 Pickle dictionaries.

    When ``cache_memmap=True`` (default), the first load writes a compact
    binary ``iq.mmap`` cache on disk and immediately frees the Pickle dict from RAM.
    All subsequent epochs and runs stream individual samples per batch via OS mmap.
    """

    def __init__(
        self,
        path: str | Path,
        name: str = "rml_pickle",
        layout: str = "channels_first",
        cache_memmap: bool = True,
        cache_dir: Optional[str | Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.layout = layout
        self.cache_memmap = bool(cache_memmap)
        self._mmap: Optional[np.memmap] = None
        self._blocks: Optional[List[np.ndarray]] = None
        self._offsets: Optional[List[int]] = None

        if self.cache_memmap:
            cdir = _memmap_cache_dir(self.path, layout, cache_dir)
            meta_f = cdir / "meta.json"
            mmap_f = cdir / "iq.mmap"
            lbl_f = cdir / "labels.npy"
            snr_f = cdir / "snrs.npy"
            if not (meta_f.is_file() and mmap_f.is_file() and lbl_f.is_file() and snr_f.is_file()):
                self._build_cache(cdir)
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            self._mmap = np.memmap(str(mmap_f), dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
            self._init_metadata(
                name=name,
                file_paths=[self.path],
                raw_labels=np.load(lbl_f),
                raw_snrs=np.load(snr_f),
                class_names=meta["classes"],
                storage_backend="pickle_memmap",
            )
        else:
            raw_dict = self._load_pickle()
            classes = sorted({str(k[0]) for k in raw_dict.keys()})
            blocks, labels, snrs, offsets = [], [], [], [0]
            for mod in classes:
                keys = sorted((k for k in raw_dict if str(k[0]) == mod), key=lambda k: float(k[1]))
                for k in keys:
                    blk = np.asarray(raw_dict[k])
                    blocks.append(blk)
                    n = len(blk)
                    labels.extend([mod] * n)
                    snrs.extend([float(k[1])] * n)
                    offsets.append(offsets[-1] + n)
            self._blocks = blocks
            self._offsets = offsets
            self._init_metadata(
                name=name,
                file_paths=[self.path],
                raw_labels=np.asarray(labels),
                raw_snrs=np.asarray(snrs, dtype=np.float32),
                class_names=classes,
                storage_backend="pickle_blocks",
            )

    def _load_pickle(self) -> Dict[Any, Any]:
        with open(self.path, "rb") as f:
            try:
                data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                data = pickle.load(f, encoding="latin1")
        if not isinstance(data, dict) or not data:
            raise ValueError(f"Expected non-empty dict in pickle file: {self.path}")
        return data

    def _build_cache(self, cdir: Path) -> None:
        cdir.mkdir(parents=True, exist_ok=True)
        raw_dict = self._load_pickle()
        classes = sorted({str(k[0]) for k in raw_dict.keys()})

        total_n = 0
        seq_len: Optional[int] = None
        entries = []
        for mod_idx, mod in enumerate(classes):
            keys = sorted((k for k in raw_dict if str(k[0]) == mod), key=lambda k: float(k[1]))
            for k in keys:
                arr = np.asarray(raw_dict[k])
                if seq_len is None:
                    seq_len = int(normalize_iq_layout(arr[0], self.layout).shape[-1])
                total_n += len(arr)
                entries.append((mod_idx, float(k[1]), k))

        assert seq_len is not None
        shape = (total_n, 2, seq_len)
        mm = np.memmap(str(cdir / "iq.mmap"), dtype=np.float32, mode="w+", shape=shape)
        labels = np.empty(total_n, dtype=np.int64)
        snrs = np.empty(total_n, dtype=np.float32)

        cursor = 0
        for mod_idx, snr_val, k in entries:
            arr = np.asarray(raw_dict[k])
            n = len(arr)
            if arr.shape[1] == 2:
                block = arr.astype(np.float32, copy=False)
            else:
                block = np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)
            mm[cursor : cursor + n] = block
            labels[cursor : cursor + n] = mod_idx
            snrs[cursor : cursor + n] = snr_val
            cursor += n

        mm.flush()
        del mm, raw_dict
        gc.collect()

        np.save(cdir / "labels.npy", labels)
        np.save(cdir / "snrs.npy", snrs)
        (cdir / "meta.json").write_text(
            json.dumps({"shape": list(shape), "classes": classes}, indent=2),
            encoding="utf-8",
        )

    def _read_iq(self, idx: int) -> np.ndarray:
        if self._mmap is not None:
            return np.array(self._mmap[idx], dtype=np.float32, copy=True)
        assert self._offsets is not None and self._blocks is not None
        part = bisect.bisect_right(self._offsets, idx) - 1
        return normalize_iq_layout(self._blocks[part][idx - self._offsets[part]], self.layout)


class LazyHDF5Dataset(BaseAMCDataset):
    """True lazy out-of-core HDF5 loader for RML2018.01a (20+ GB) and HisarMod2019.h5.

    Design guarantees:
        1. ``X`` (IQ waveform tensor) is NEVER loaded into RAM during ``__init__``.
        2. Only lightweight ``Y`` (labels) and ``Z`` (SNR) are scanned in 8,192-sample chunks.
        3. Inside ``__getitem__(idx)``, only the single requested slice ``X[local_idx]`` is read
           from disk and transposed from ``[L, 2]`` to ``[2, L]`` if needed.
        4. Multi-worker DataLoader safe: tracks ``os.getpid()`` and re-opens read-only HDF5
           file handles per worker process; ``__getstate__`` strips handles before pickling.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        name: str = "rml2018",
        layout: str = "auto",
        keys: Optional[Dict[str, str]] = None,
        class_names: Optional[Sequence[str]] = None,
        file_splits: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading HDF5 datasets requires h5py. Install with: pip install h5py") from exc

        if isinstance(paths, (str, Path)):
            path_list = [Path(paths).expanduser().resolve()]
        else:
            path_list = [Path(p).expanduser().resolve() for p in paths]
        for p in path_list:
            if not p.is_file():
                raise FileNotFoundError(f"HDF5 dataset file not found: {p}")

        self._paths = [str(p) for p in path_list]
        self.layout = layout
        self._handles: Optional[List[Any]] = None
        self._owner_pid: Optional[int] = None

        req_keys = keys or {"x": "X", "y": "Y", "snr": "Z"}
        self._resolved_x_keys: List[str] = []
        lengths: List[int] = []
        y_parts: List[np.ndarray] = []
        z_parts: List[np.ndarray] = []

        for p_str in self._paths:
            with h5py.File(p_str, "r") as f:
                xk = req_keys.get("x", "X")
                yk = req_keys.get("y", "Y")
                zk = req_keys.get("snr", "Z")
                xk = xk if xk in f else next((k for k in ("X", "samples", "data", "iq") if k in f), xk)
                yk = yk if yk in f else next((k for k in ("Y", "labels", "label") if k in f), yk)
                zk = zk if zk in f else next((k for k in ("Z", "snr", "snrs") if k in f), zk)
                self._resolved_x_keys.append(xk)

                x_ds, y_ds = f[xk], f[yk]
                n = len(x_ds)
                if len(y_ds) != n:
                    raise ValueError(f"Sample count mismatch in {p_str}: {xk}={n}, {yk}={len(y_ds)}")
                lengths.append(n)

                # Chunked scan of Y and Z only (never touch X here)
                yy, zz = [], []
                for start in range(0, n, 8192):
                    end = min(start + 8192, n)
                    yy.append(labels_to_1d(y_ds[start:end]))
                    if zk in f:
                        zz.append(np.asarray(f[zk][start:end], dtype=np.float32).reshape(-1))
                y_parts.append(np.concatenate(yy) if yy else np.empty(0, dtype=np.int64))
                z_parts.append(np.concatenate(zz) if zz else np.zeros(n, dtype=np.float32))

        self._offsets = np.cumsum([0] + lengths).tolist()
        all_labels = np.concatenate(y_parts)
        all_snrs = np.concatenate(z_parts)

        if class_names is None:
            n_uniq = len(np.unique(all_labels))
            if "2018" in name.lower() and n_uniq == 24:
                class_names = RML2018_CLASSES
            elif "hisar" in name.lower() and n_uniq == 26:
                class_names = HISARMOD_CLASSES

        self._init_metadata(
            name=name,
            file_paths=self._paths,
            raw_labels=all_labels,
            raw_snrs=all_snrs,
            class_names=class_names,
            storage_backend="hdf5_lazy",
        )
        # Ensure no open HDF5 handle is held after initialization
        self.close()

        if file_splits and "test" in file_splits:
            test_indices: List[int] = []
            for i, sp in enumerate(file_splits):
                if sp == "test":
                    test_indices.extend(range(self._offsets[i], self._offsets[i + 1]))
            self.predefined_test_indices = np.asarray(test_indices, dtype=np.int64)

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._handles is not None and self._owner_pid != pid:
            self.close()
        if self._handles is None:
            import h5py
            self._handles = [h5py.File(p, "r", swmr=True) for p in self._paths]
            self._owner_pid = pid

    def _read_iq(self, idx: int) -> np.ndarray:
        self._ensure_open()
        assert self._handles is not None
        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[file_idx]
        raw_sample = self._handles[file_idx][self._resolved_x_keys[file_idx]][local_idx]
        return normalize_iq_layout(raw_sample, self.layout)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = None
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        if self._handles is not None:
            for h in self._handles:
                try:
                    h.close()
                except Exception:
                    pass
            self._handles = None
            self._owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class NpzMmapDataset(BaseAMCDataset):
    """Memory-mapped ``.npz`` dataset loader."""

    def __init__(
        self,
        path: str | Path,
        name: str = "npz_dataset",
        layout: str = "auto",
        keys: Optional[Dict[str, str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        self.path = Path(path).expanduser().resolve()
        self.layout = layout
        k = keys or {"x": "X", "y": "Y", "snr": "Z"}
        self._npz = np.load(self.path, allow_pickle=False, mmap_mode="r")
        self._x = self._npz[k.get("x", "X")]
        raw_y = self._npz[k.get("y", "Y")]
        raw_z = self._npz[k.get("snr", "Z")] if k.get("snr", "Z") in self._npz else None
        names = self._npz["class_names"].tolist() if "class_names" in self._npz else None
        self._init_metadata(name, [self.path], raw_y, raw_z, names, storage_backend="npz_mmap")

    def _read_iq(self, idx: int) -> np.ndarray:
        return normalize_iq_layout(self._x[idx], self.layout)


class SyntheticAMCDataset(BaseAMCDataset):
    """Lightweight synthetic IQ dataset for instant smoke tests and prototyping."""

    def __init__(
        self,
        num_classes: int = 11,
        seq_length: int = 128,
        snr_values: Sequence[int] = (-10, -4, 0, 6, 12, 18),
        samples_per_cell: int = 15,
        seed: int = 42,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(transform=transform)
        rng = np.random.default_rng(seed)
        default_names = [
            "8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
            "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM",
        ]
        names = default_names[:num_classes] if num_classes <= len(default_names) else [f"MOD_{i:02d}" for i in range(num_classes)]
        t = np.arange(seq_length, dtype=np.float32)

        xs, ys, zs = [], [], []
        for c in range(num_classes):
            freq = 0.02 + 0.014 * c
            for snr in snr_values:
                noise_std = 0.35 * (10.0 ** (-snr / 20.0))
                for _ in range(samples_per_cell):
                    phi0 = float(rng.uniform(-np.pi, np.pi))
                    sig = np.exp(1j * (2.0 * np.pi * freq * t + phi0))
                    noise = rng.normal(0.0, noise_std, size=seq_length) + 1j * rng.normal(0.0, noise_std, size=seq_length)
                    v = sig + noise
                    xs.append(np.stack((v.real, v.imag), axis=0).astype(np.float32))
                    ys.append(c)
                    zs.append(float(snr))

        self._x = np.stack(xs, axis=0)
        self._init_metadata(
            name=f"synthetic_L{seq_length}",
            file_paths=[],
            raw_labels=np.array(ys, dtype=np.int64),
            raw_snrs=np.array(zs, dtype=np.float32),
            class_names=names,
            storage_backend="synthetic_memory",
        )

    def _read_iq(self, idx: int) -> np.ndarray:
        return self._x[idx]


def load_dataset(
    name: str,
    path: Optional[str | Path] = None,
    split: Optional[Tuple[float, float, float]] = (0.6, 0.2, 0.2),
    seed: int = 42,
    split_file: Optional[str | Path] = None,
    train_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eval_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cache_memmap: bool = True,
    **kwargs: Any,
) -> Any:
    """High-level entry point to load any supported AMC dataset and optionally split it.

    Args:
        name: One of ``"rml2016.10a"``, ``"rml2016.10b"``, ``"rml22"``,
            ``"rml2018"``, ``"hisarmod2019"``, or ``"synthetic"``.
        path: File or directory path to the dataset (not required for ``"synthetic"``).
        split: ``(train, val, test)`` ratios for joint ``(modulation, SNR)`` stratified
            splitting. Pass ``None`` to return the full unsplit ``BaseAMCDataset``.
        seed: Random seed controlling deterministic stratified splits.
        split_file: Optional ``.npz`` path to save or reuse exact train/val/test indices.
            If the file exists, indices are loaded from it; otherwise the computed split
            is saved to this path for guaranteed reuse across models and runs.
        train_transform: Optional transform/augmentation applied ONLY to the train split.
        eval_transform: Optional transform applied to val and test splits.
        cache_memmap: If True, Pickle datasets are cached to disk as ``np.memmap`` for
            out-of-core per-batch streaming.

    Returns:
        ``(train_ds, val_ds, test_ds)`` if ``split`` is provided, else ``BaseAMCDataset``.
    """
    key = name.strip().lower().replace("_", ".")

    if key == "synthetic":
        ds: BaseAMCDataset = SyntheticAMCDataset(
            num_classes=int(kwargs.get("num_classes", 11)),
            seq_length=int(kwargs.get("seq_length", 128)),
            snr_values=kwargs.get("snr_values", (-10, -4, 0, 6, 12, 18)),
            samples_per_cell=int(kwargs.get("samples_per_cell", 15)),
            seed=seed,
        )
    elif key in ("rml2016.10a", "rml2016a", "rml2016.10b", "rml2016b", "rml22", "rml22.01a"):
        if not path:
            raise ValueError(f"Dataset '{name}' requires a valid file 'path'.")
        ds = RMLPickleDataset(
            path=path,
            name=name,
            layout=kwargs.get("layout", "channels_first"),
            cache_memmap=cache_memmap,
            cache_dir=kwargs.get("cache_dir"),
        )
    elif key in ("rml2018", "rml2018.01a"):
        if not path:
            raise ValueError("Dataset 'rml2018' requires the path to GOLD_XYZ_OSC.0001_1024.hdf5.")
        ds = LazyHDF5Dataset(
            paths=path,
            name="rml2018",
            layout=kwargs.get("layout", "channels_last"),
            keys=kwargs.get("keys", {"x": "X", "y": "Y", "snr": "Z"}),
        )
    elif key in ("hisarmod2019", "hisarmod2019.1", "hizarmod2019"):
        if not path:
            raise ValueError("Dataset 'hisarmod2019' requires a file or directory 'path'.")
        p = Path(path).expanduser().resolve()
        if p.is_dir():
            train_h5 = p / "HisarMod2019train.h5"
            test_h5 = p / "HisarMod2019test.h5"
            if train_h5.is_file() and test_h5.is_file():
                ds = LazyHDF5Dataset(
                    paths=[train_h5, test_h5],
                    name="hisarmod2019",
                    layout=kwargs.get("layout", "auto"),
                    keys=kwargs.get("keys", {"x": "samples", "y": "labels", "snr": "snr"}),
                    file_splits=["train", "test"],
                )
            else:
                h5_files = sorted(list(p.glob("*.h5")) + list(p.glob("*.hdf5")))
                if not h5_files:
                    raise FileNotFoundError(f"No .h5/.hdf5 files found in {p}")
                ds = LazyHDF5Dataset(paths=h5_files, name="hisarmod2019", layout=kwargs.get("layout", "auto"))
        elif p.suffix.lower() in (".h5", ".hdf5"):
            ds = LazyHDF5Dataset(paths=p, name="hisarmod2019", layout=kwargs.get("layout", "auto"))
        elif p.suffix.lower() == ".npz":
            ds = NpzMmapDataset(path=p, name="hisarmod2019", layout=kwargs.get("layout", "auto"))
        else:
            ds = RMLPickleDataset(path=p, name="hisarmod2019", layout=kwargs.get("layout", "auto"), cache_memmap=cache_memmap)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: rml2016.10a, rml2016.10b, rml22, rml2018, hisarmod2019, synthetic."
        )

    if split is None:
        ds.transform = eval_transform or train_transform
        return ds

    if split_file is not None and Path(split_file).expanduser().is_file():
        tr_idx, va_idx, te_idx = load_split_indices(split_file, len(ds))
    else:
        tr_idx, va_idx, te_idx = stratified_split_indices(
            labels=ds.labels,
            snrs=ds.snrs,
            split=split,
            seed=seed,
            predefined_test=ds.predefined_test_indices,
        )
        if split_file is not None:
            save_split_indices(split_file, tr_idx, va_idx, te_idx)
    train_ds = AMCSubset(ds, tr_idx, transform=train_transform)
    val_ds = AMCSubset(ds, va_idx, transform=eval_transform)
    test_ds = AMCSubset(ds, te_idx, transform=eval_transform)
    return train_ds, val_ds, test_ds
