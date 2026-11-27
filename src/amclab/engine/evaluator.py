"""Keras-like Evaluator for clean model evaluation and per-SNR reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from amclab.engine.metrics import compute_amc_metrics
from amclab.utils.runtime import resolve_device


class Evaluator:
    """Clean, Keras-style evaluator for any PyTorch AMC model ``model(iq) -> logits``.

    Example::

        evaluator = Evaluator(model, device="auto")
        metrics = evaluator.evaluate(test_ds, batch_size=256, checkpoint="./runs/exp1/checkpoints/best.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        criterion: Optional[nn.Module] = None,
        show_progress: bool = True,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.show_progress = bool(show_progress)

    def load_weights(self, checkpoint_path: str | Path) -> None:
        """Load model weights from either a Trainer checkpoint dict or raw state_dict."""
        p = Path(checkpoint_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")
        state = torch.load(p, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        else:
            self.model.load_state_dict(state)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 256,
        num_workers: int = 0,
        checkpoint: Optional[str | Path] = None,
        save_dir: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the model on ``dataset`` and optionally persist metrics/CSV/CM."""
        if checkpoint is not None:
            self.load_weights(checkpoint)

        pin_memory = self.device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        num_classes = int(getattr(dataset, "num_classes", 0))
        class_names = list(getattr(dataset, "class_names", []))

        self.model.eval()
        loss_sum, total = 0.0, 0
        all_y, all_pred, all_snr = [], [], []

        pbar = tqdm(loader, desc="Evaluating", leave=False, disable=not self.show_progress, dynamic_ncols=True)
        for iq, labels, snrs in pbar:
            iq = iq.to(self.device, non_blocking=pin_memory)
            labels = labels.to(self.device, non_blocking=pin_memory)
            logits = self.model(iq)
            if num_classes == 0:
                num_classes = int(logits.shape[1])
            loss = self.criterion(logits, labels)

            bs = labels.size(0)
            loss_sum += float(loss.item()) * bs
            total += bs
            preds = logits.argmax(dim=1)

            all_y.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_snr.append(snrs.numpy())

        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_pred)
        z_snr = np.concatenate(all_snr)
        metrics = compute_amc_metrics(y_true, y_pred, z_snr, num_classes, loss_sum / max(total, 1))
        if not class_names:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        metrics["class_names"] = class_names

        if verbose:
            print(
                f"[Eval] Loss: {metrics['loss']:.4f} | "
                f"Overall Acc: {metrics['overall_accuracy']*100:.2f}% | "
                f"High-SNR (>=0dB) Acc: {metrics['high_snr_accuracy']*100:.2f}% | "
                f"Macro-F1: {metrics['macro_f1']*100:.2f}% | "
                f"Kappa: {metrics['kappa']:.4f}"
            )

        if save_dir is not None:
            self._save_results(Path(save_dir), metrics)

        return metrics

    @staticmethod
    def _save_results(out_dir: Path, metrics: Dict[str, Any]) -> None:
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        per_snr = metrics.get("per_snr_accuracy", [])
        if per_snr:
            with open(out_dir / "snr_accuracy.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["snr_db", "samples", "accuracy", "macro_f1"])
                w.writeheader()
                w.writerows(per_snr)

        cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        names = metrics["class_names"]
        with open(out_dir / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for name, row in zip(names, cm.tolist()):
                w.writerow([name] + row)
