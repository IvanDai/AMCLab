"""Classification and per-SNR evaluation metrics for AMC tasks."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _stats_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    total = max(int(cm.sum()), 1)
    acc = float(np.trace(cm) / total)
    tp = cm.diagonal().astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    precision = tp / np.maximum(1.0, cm.sum(axis=0))
    recall = tp / np.maximum(1.0, support)
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)

    valid = support > 0
    macro_f1 = float(f1[valid].mean()) if np.any(valid) else 0.0
    weighted_f1 = float(np.sum(f1 * support) / total)
    pe = float(np.sum(cm.sum(axis=0) * support)) / float(total * total)
    kappa = float((acc - pe) / max(1.0 - pe, 1e-12)) if pe < 1.0 else 1.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "per_class_f1": f1.tolist(),
    }


def compute_amc_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    snrs: Sequence[float] | np.ndarray,
    num_classes: int,
    avg_loss: float,
) -> Dict[str, Any]:
    """Compute overall, high-SNR, macro-F1, kappa, per-SNR accuracy, and confusion matrix."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    zs = np.asarray(snrs, dtype=np.float32)

    cm = np.bincount(yt * num_classes + yp, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    base = _stats_from_cm(cm)

    per_snr = []
    high_snr_accs = []
    for snr in sorted(np.unique(zs[np.isfinite(zs)])):
        mask = np.isclose(zs, snr)
        sub_cm = np.bincount(yt[mask] * num_classes + yp[mask], minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
        sub = _stats_from_cm(sub_cm)
        per_snr.append({
            "snr_db": float(snr),
            "samples": int(mask.sum()),
            "accuracy": sub["accuracy"],
            "macro_f1": sub["macro_f1"],
        })
        if snr >= 0.0:
            high_snr_accs.append(sub["accuracy"])

    return {
        "loss": float(avg_loss),
        "overall_accuracy": base["accuracy"],
        "high_snr_accuracy": float(np.mean(high_snr_accs)) if high_snr_accs else base["accuracy"],
        "macro_f1": base["macro_f1"],
        "weighted_f1": base["weighted_f1"],
        "kappa": base["kappa"],
        "per_class_f1": base["per_class_f1"],
        "num_samples": int(len(yt)),
        "per_snr_accuracy": per_snr,
        "confusion_matrix": cm.tolist(),
    }
