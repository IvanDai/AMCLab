#!/usr/bin/env python3
"""Example 01: Train a single built-in baseline, resume for additional epochs, and evaluate.

Run from anywhere after `pip install -e .`:
    python examples/01_single_baseline_train_resume/train_and_resume.py
"""

from pathlib import Path
import amclab
from amclab import Evaluator, Trainer, get_model, load_dataset
from amclab.transforms import Compose, RandomPhaseRotation, UnitPowerNormalize


# ==============================================================================
# DATASET & EXPERIMENT CONFIGURATION (Edit these paths for your real dataset)
# ==============================================================================
DATASET_NAME = "synthetic"  # Options: "rml2016.10a", "rml2016.10b", "rml22", "rml2018", "hisarmod2019", "synthetic"
DATASET_PATH = None         # e.g., "/data/AMC/RML2016.10a_dict.pkl" or "/data/AMC/GOLD_XYZ_OSC.0001_1024.hdf5"
SPLIT_FILE = "./runs/example01_single/splits_seed42.npz"  # Saved on first run and reused on resume/eval

MODEL_NAME = "mcldnn"       # Any name from amclab.list_models()
SAVE_DIR = Path("./runs/example01_single") / MODEL_NAME
PHASE1_EPOCHS = 3
EXTRA_EPOCHS = 2
BATCH_SIZE = 32
SEED = 42
# ==============================================================================


def main() -> None:
    print("Environment:", amclab.doctor())

    # 1. Load dataset with persisted split_file to guarantee identical splits across phases
    train_ds, val_ds, test_ds = load_dataset(
        name=DATASET_NAME,
        path=DATASET_PATH,
        split=(0.6, 0.2, 0.2),
        seed=SEED,
        split_file=SPLIT_FILE,
        train_transform=Compose([UnitPowerNormalize(), RandomPhaseRotation(p=0.5)]),
        eval_transform=UnitPowerNormalize(),
    )

    # 2. Instantiate a built-in baseline from amclab
    model = get_model(
        MODEL_NAME,
        num_classes=train_ds.num_classes,
        seq_length=train_ds.seq_length,
    )

    # 3. Phase 1: Initial training (default lr_scheduler="plateau" with factor=0.5, patience=3, min_lr=1e-6)
    #    You can also pass a torch.optim.lr_scheduler instance/class or callable, e.g.:
    #    lr_scheduler=lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    print(f"\n>>> Phase 1: Training {MODEL_NAME} for {PHASE1_EPOCHS} epochs")
    trainer = Trainer(
        model=model,
        optimizer="adamw",
        lr=1e-3,
        lr_scheduler="plateau",
        device="auto",
        save_dir=SAVE_DIR,
        seed=SEED,
    )
    trainer.fit(train_ds, val_ds, epochs=PHASE1_EPOCHS, batch_size=BATCH_SIZE)

    # 4. Phase 2: Resume / Additional training (+EXTRA_EPOCHS from last.pt; history is appended)
    print(f"\n>>> Phase 2: Resuming {MODEL_NAME} for +{EXTRA_EPOCHS} extra epochs")
    trainer.fit(
        train_ds,
        val_ds,
        extra_epochs=EXTRA_EPOCHS,
        batch_size=BATCH_SIZE,
        resume_from=SAVE_DIR / "checkpoints/last.pt",
    )

    # 5. Evaluate best checkpoint on test split
    evaluator = Evaluator(model=model, device="auto")
    evaluator.evaluate(
        test_ds,
        batch_size=BATCH_SIZE * 2,
        checkpoint=SAVE_DIR / "checkpoints/best.pt",
        save_dir=SAVE_DIR / "eval",
    )


if __name__ == "__main__":
    main()
