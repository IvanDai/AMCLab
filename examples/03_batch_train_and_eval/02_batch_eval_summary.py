#!/usr/bin/env python3
"""Example 03 - Script 2: Batch load trained checkpoints, evaluate on test set, and export summary tables.

Run after 01_batch_train.py:
    python examples/03_batch_train_and_eval/02_batch_eval_summary.py
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

from amclab import Evaluator, get_model, load_dataset
from amclab.transforms import UnitPowerNormalize
from amclab.utils import count_parameters

from config import (
    DATASET_NAME,
    DATASET_PATH,
    EVAL_BATCH_SIZE,
    MODELS_TO_RUN,
    OUTPUT_ROOT,
    SEED,
    SPLIT_FILE,
)


def main() -> None:
    # Reuse the exact same SPLIT_FILE generated during training
    _, _, test_ds = load_dataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        split=(0.6, 0.2, 0.2),
        seed=SEED,
        split_file=SPLIT_FILE,
        eval_transform=UnitPowerNormalize(),
    )

    summary_rows: List[Dict[str, Any]] = []
    per_snr_records: Dict[str, Dict[float, float]] = {}
    all_snrs = set()

    for model_name in MODELS_TO_RUN:
        ckpt_path = OUTPUT_ROOT / model_name / "checkpoints" / "best.pt"
        if not ckpt_path.is_file():
            print(f"[Skip] Checkpoint not found for '{model_name}': {ckpt_path}")
            continue

        model = get_model(
            model_name,
            num_classes=test_ds.num_classes,
            seq_length=test_ds.seq_length,
        )
        num_params = count_parameters(model)
        evaluator = Evaluator(model=model, device="auto")
        metrics = evaluator.evaluate(
            test_ds,
            batch_size=EVAL_BATCH_SIZE,
            checkpoint=ckpt_path,
            save_dir=OUTPUT_ROOT / model_name / "eval",
            verbose=False,
        )

        summary_rows.append({
            "model": model_name,
            "params": num_params,
            "test_loss": round(metrics["loss"], 4),
            "overall_acc_pct": round(metrics["overall_accuracy"] * 100.0, 2),
            "high_snr_acc_pct": round(metrics["high_snr_accuracy"] * 100.0, 2),
            "macro_f1_pct": round(metrics["macro_f1"] * 100.0, 2),
            "kappa": round(metrics["kappa"], 4),
        })

        snr_map = {}
        for item in metrics.get("per_snr_accuracy", []):
            s = float(item["snr_db"])
            all_snrs.add(s)
            snr_map[s] = round(item["accuracy"] * 100.0, 2)
        per_snr_records[model_name] = snr_map

    # Sort summary table by overall accuracy descending
    summary_rows.sort(key=lambda r: r["overall_acc_pct"], reverse=True)

    # 1. Print formatted table to console
    print("\n" + "=" * 86)
    print(f"{'Model':<14} | {'Params':>10} | {'OA (%)':>8} | {'High-SNR (%)':>12} | {'Macro-F1 (%)':>12} | {'Kappa':>8}")
    print("-" * 86)
    for row in summary_rows:
        print(
            f"{row['model']:<14} | {row['params']:>10,} | {row['overall_acc_pct']:>8.2f} | "
            f"{row['high_snr_acc_pct']:>12.2f} | {row['macro_f1_pct']:>12.2f} | {row['kappa']:>8.4f}"
        )
    print("=" * 86)

    # 2. Save overall summary CSV
    summary_csv = OUTPUT_ROOT / "benchmark_summary.csv"
    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved overall benchmark table to: {summary_csv.resolve()}")

    # 3. Save per-SNR accuracy matrix CSV
    sorted_snrs = sorted(all_snrs)
    if sorted_snrs and per_snr_records:
        per_snr_csv = OUTPUT_ROOT / "benchmark_per_snr.csv"
        header = ["model"] + [f"{int(s) if s.is_integer() else s}dB" for s in sorted_snrs]
        with open(per_snr_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in summary_rows:
                m = row["model"]
                smap = per_snr_records.get(m, {})
                writer.writerow([m] + [smap.get(s, "") for s in sorted_snrs])
        print(f"Saved per-SNR accuracy table to: {per_snr_csv.resolve()}")


if __name__ == "__main__":
    main()
