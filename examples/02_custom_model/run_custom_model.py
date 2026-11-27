#!/usr/bin/env python3
"""Example 02: Train, resume, and evaluate custom model variants defined in a separate file."""

from pathlib import Path
from amclab import Evaluator, Trainer, load_dataset
from amclab.transforms import UnitPowerNormalize

from my_model import MyCustomAMCNet


# ==============================================================================
# DATASET CONFIGURATION (Edit these paths for your real dataset)
# ==============================================================================
DATASET_NAME = "synthetic"  # e.g., "rml2016.10a" or "rml2018"
DATASET_PATH = None         # e.g., "/data/AMC/RML2016.10a_dict.pkl"
SPLIT_FILE = "./runs/example02_custom/splits_seed42.npz"
OUTPUT_ROOT = Path("./runs/example02_custom")
SEED = 42
# ==============================================================================


def main() -> None:
    train_ds, val_ds, test_ds = load_dataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        split=(0.6, 0.2, 0.2),
        seed=SEED,
        split_file=SPLIT_FILE,
        train_transform=UnitPowerNormalize(),
        eval_transform=UnitPowerNormalize(),
    )

    variants = {
        "my_net_light": MyCustomAMCNet(num_classes=train_ds.num_classes, width=32, num_blocks=2),
        "my_net_deep": MyCustomAMCNet(num_classes=train_ds.num_classes, width=64, num_blocks=4),
    }

    for name, model in variants.items():
        save_dir = OUTPUT_ROOT / name
        print(f"\n>>> Training custom variant: {name}")
        trainer = Trainer(
            model=model,
            optimizer="adamw",
            lr=1e-3,
            scheduler="cosine",
            device="auto",
            save_dir=save_dir,
            seed=SEED,
        )
        trainer.fit(train_ds, val_ds, epochs=3, batch_size=32)

        # Resume for 2 extra epochs
        trainer.fit(
            train_ds,
            val_ds,
            extra_epochs=2,
            batch_size=32,
            resume_from=save_dir / "checkpoints/last.pt",
        )

        evaluator = Evaluator(model=model, device="auto")
        evaluator.evaluate(
            test_ds,
            batch_size=64,
            checkpoint=save_dir / "checkpoints/best.pt",
            save_dir=save_dir / "eval",
        )


if __name__ == "__main__":
    main()
