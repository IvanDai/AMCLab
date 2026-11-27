"""Keras-like Trainer for clean, boilerplate-free AMC training and seamless resumption."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from amclab.engine.evaluator import Evaluator
from amclab.utils.runtime import (
    count_parameters,
    describe_device,
    get_rng_state,
    resolve_device,
    seed_everything,
    set_rng_state,
)


class Trainer:
    """Keras-style PyTorch Trainer tailored for Automatic Modulation Classification.

    Features:
        - Auto-detects NVIDIA CUDA, Apple MPS, or CPU and adapts ``pin_memory`` / AMP.
        - Live ``tqdm`` progress bars displaying loss, accuracy, learning rate, and samples/sec.
        - Atomic checkpoint saving (``best.pt`` and ``last.pt``) with optimizer, scheduler,
          and RNG state for exact resumption or additional epochs (``extra_epochs``).

    Example::

        trainer = Trainer(model, optimizer="adamw", lr=1e-3, scheduler="cosine", save_dir="./runs/exp1")
        history = trainer.fit(train_ds, val_ds, epochs=50, batch_size=128)
        # Continue training for 20 more epochs:
        trainer.fit(train_ds, val_ds, extra_epochs=20, resume_from="./runs/exp1/checkpoints/last.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: str | torch.optim.Optimizer = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: Optional[str] = "cosine",
        loss_fn: Optional[nn.Module] = None,
        label_smoothing: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        use_amp: bool = True,
        device: str | torch.device = "auto",
        save_dir: Optional[str | Path] = None,
        seed: int = 42,
        show_progress: bool = True,
    ):
        seed_everything(seed)
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.scheduler_name = (scheduler or "none").lower() if isinstance(scheduler, str) else "custom"
        self.custom_scheduler = None if isinstance(scheduler, str) or scheduler is None else scheduler
        self.criterion = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        self.grad_clip = grad_clip
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.show_progress = bool(show_progress)
        self.seed = int(seed)

        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
        if self.save_dir:
            (self.save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_optimizer(optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: List[Dict[str, Any]] = []
        self.best_val_acc: float = -1.0
        self.best_epoch: int = 0

    def _build_optimizer(self, opt: str | torch.optim.Optimizer) -> torch.optim.Optimizer:
        if isinstance(opt, torch.optim.Optimizer):
            return opt
        name = opt.strip().lower()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer '{opt}'. Choose from: adamw, adam, sgd.")

    def _build_scheduler(self, total_epochs: int):
        if self.custom_scheduler is not None:
            return self.custom_scheduler
        if self.scheduler_name in ("none", "null", ""):
            return None
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(total_epochs, 1), eta_min=1e-6)
        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, total_epochs // 3), gamma=0.5)
        if self.scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)
        raise ValueError(f"Unsupported scheduler '{self.scheduler_name}'. Choose from: cosine, step, plateau, none.")

    def _save_checkpoint(self, filename: str, epoch: int, scheduler: Any, bad_epochs: int) -> Optional[Path]:
        if self.save_dir is None:
            return None
        path = self.save_dir / "checkpoints" / filename
        payload = {
            "epoch": int(epoch),
            "best_val_acc": float(self.best_val_acc),
            "best_epoch": int(self.best_epoch),
            "bad_epochs": int(bad_epochs),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
            "rng_state": get_rng_state(),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def _save_history(self) -> None:
        if self.save_dir is None or not self.history:
            return
        (self.save_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        with open(self.save_dir / "history.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: Optional[int] = 50,
        extra_epochs: Optional[int] = None,
        batch_size: int = 128,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 0,
        early_stopping: Optional[int] = 15,
        resume_from: Optional[str | Path] = None,
    ) -> List[Dict[str, Any]]:
        """Train (or resume training) the model on ``train_dataset`` with optional validation.

        Args:
            train_dataset: PyTorch Dataset returning ``(iq, label, snr)``.
            val_dataset: Optional validation Dataset.
            epochs: Total target epochs (e.g., ``epochs=50``).
            extra_epochs: When resuming, train this many additional epochs beyond the checkpoint epoch.
            batch_size: Training mini-batch size.
            eval_batch_size: Validation batch size (defaults to ``2 * batch_size``).
            num_workers: Number of DataLoader worker processes.
            early_stopping: Stop if validation accuracy does not improve for this many epochs (None to disable).
            resume_from: Path to a saved ``last.pt`` or ``best.pt`` checkpoint to continue training from.
        """
        pin_memory = self.device.type == "cuda"
        eval_bs = eval_batch_size or (batch_size * 2)

        start_epoch = 1
        bad_epochs = 0
        ckpt_state = None
        if resume_from is not None:
            ckpt_path = Path(resume_from).expanduser().resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            ckpt_state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt_state["model_state_dict"])
            if ckpt_state.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            if ckpt_state.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(ckpt_state["scaler_state_dict"])
            start_epoch = int(ckpt_state.get("epoch", 0)) + 1
            self.best_val_acc = float(ckpt_state.get("best_val_acc", -1.0))
            self.best_epoch = int(ckpt_state.get("best_epoch", 0))
            bad_epochs = int(ckpt_state.get("bad_epochs", 0))
            self.history = list(ckpt_state.get("history", []))
            set_rng_state(ckpt_state.get("rng_state"))

        if extra_epochs is not None:
            target_epochs = (start_epoch - 1) + int(extra_epochs)
        elif epochs is not None:
            target_epochs = int(epochs)
        else:
            raise ValueError("Please provide either 'epochs' or 'extra_epochs'.")

        scheduler = self._build_scheduler(target_epochs)
        if ckpt_state is not None and scheduler is not None and ckpt_state.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
            except Exception:
                pass

        evaluator = Evaluator(self.model, device=self.device, criterion=self.criterion, show_progress=self.show_progress)

        print(
            f"[AMCLab] Model: {self.model.__class__.__name__} ({count_parameters(self.model):,} params) | "
            f"Device: {describe_device(self.device)} | AMP: {self.use_amp}"
        )
        if resume_from is not None:
            print(f"[AMCLab] Resuming from epoch {start_epoch - 1} -> target epoch {target_epochs} ({resume_from})")
        if self.save_dir is not None:
            print(f"[AMCLab] Output directory: {self.save_dir}")

        for epoch in range(start_epoch, target_epochs + 1):
            t0 = time.time()
            gen = torch.Generator()
            gen.manual_seed(self.seed + epoch * 100003)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                generator=gen,
            )

            self.model.train()
            tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch:03d}/{target_epochs:03d}] Train",
                leave=False,
                disable=not self.show_progress,
                dynamic_ncols=True,
            )

            for iq, labels, _ in pbar:
                iq = iq.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    logits = self.model(iq)
                    loss = self.criterion(logits, labels)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                    self.optimizer.step()

                bs = labels.size(0)
                tr_loss_sum += float(loss.detach().item()) * bs
                tr_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
                tr_total += bs
                elapsed = max(time.time() - t0, 1e-6)
                pbar.set_postfix(
                    loss=f"{tr_loss_sum / tr_total:.4f}",
                    acc=f"{(tr_correct / tr_total) * 100:.2f}%",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    ips=f"{tr_total / elapsed:.0f}",
                )

            train_loss = tr_loss_sum / max(tr_total, 1)
            train_acc = tr_correct / max(tr_total, 1)

            val_loss, val_acc, val_f1 = float("nan"), float("nan"), float("nan")
            if val_dataset is not None:
                val_res = evaluator.evaluate(
                    val_dataset, batch_size=eval_bs, num_workers=num_workers, verbose=False
                )
                val_loss = val_res["loss"]
                val_acc = val_res["overall_accuracy"]
                val_f1 = val_res["macro_f1"]

            cur_lr = float(self.optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc if val_dataset is not None else train_acc)
                else:
                    scheduler.step()

            score = val_acc if val_dataset is not None else train_acc
            is_best = score > self.best_val_acc
            if is_best:
                self.best_val_acc = score
                self.best_epoch = epoch
                bad_epochs = 0
                self._save_checkpoint("best.pt", epoch, scheduler, bad_epochs)
            else:
                bad_epochs += 1

            self._save_checkpoint("last.pt", epoch, scheduler, bad_epochs)

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6) if val_dataset is not None else None,
                "val_acc": round(val_acc, 6) if val_dataset is not None else None,
                "val_macro_f1": round(val_f1, 6) if val_dataset is not None else None,
                "lr": cur_lr,
                "time_sec": round(time.time() - t0, 2),
            }
            self.history.append(row)
            self._save_history()

            tag = " * [BEST]" if is_best else ""
            if val_dataset is not None:
                print(
                    f"Epoch {epoch:03d}/{target_epochs:03d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:6.2f}% F1: {val_f1*100:6.2f}% | "
                    f"LR: {cur_lr:.2e} | {row['time_sec']:.1f}s{tag}"
                )
            else:
                print(
                    f"Epoch {epoch:03d}/{target_epochs:03d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | "
                    f"LR: {cur_lr:.2e} | {row['time_sec']:.1f}s{tag}"
                )

            if early_stopping and bad_epochs >= int(early_stopping):
                print(f"[AMCLab] Early stopping triggered after {bad_epochs} epochs without improvement (best epoch: {self.best_epoch}).")
                break

        return self.history
