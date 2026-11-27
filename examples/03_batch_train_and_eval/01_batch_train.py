#!/usr/bin/env python3
"""Example 03 - Script 1: Batch train all built-in baseline models on the same dataset split.

Run:
    python examples/03_batch_train_and_eval/01_batch_train.py
"""

from amclab import Trainer, get_model, load_dataset
from amclab.transforms import UnitPowerNormalize

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    DATASET_PATH,
    EPOCHS,
    LR,
    MODELS_TO_RUN,
    OUTPUT_ROOT,
    SEED,
    SPLIT_FILE,
)


def main() -> None:
    # Load dataset once and persist SPLIT_FILE so every model uses the exact same train/val/test split
    train_ds, val_ds, _ = load_dataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        split=(0.6, 0.2, 0.2),
        seed=SEED,
        split_file=SPLIT_FILE,
        train_transform=UnitPowerNormalize(),
        eval_transform=UnitPowerNormalize(),
    )

    print(f"[Batch Train] Dataset: {DATASET_NAME} | Models ({len(MODELS_TO_RUN)}): {MODELS_TO_RUN}")
    print(f"[Batch Train] Shared split file: {SPLIT_FILE.resolve()}")

    for idx, model_name in enumerate(MODELS_TO_RUN, 1):
        save_dir = OUTPUT_ROOT / model_name
        print(f"\n================ [{idx}/{len(MODELS_TO_RUN)}] Training: {model_name} ================")
        model = get_model(
            model_name,
            num_classes=train_ds.num_classes,
            seq_length=train_ds.seq_length,
        )
        trainer = Trainer(
            model=model,
            optimizer="adamw",
            lr=LR,
            lr_scheduler="plateau",
            device="auto",
            save_dir=save_dir,
            seed=SEED,
        )
        trainer.fit(
            train_ds,
            val_ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            early_stopping=15,
        )

    print(f"\n[Batch Train] Finished training all {len(MODELS_TO_RUN)} models. Checkpoints saved under: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
