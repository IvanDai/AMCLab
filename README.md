# AMCLab

**A lightweight PyTorch toolkit for Automatic Modulation Classification (AMC).**

English | [简体中文](docs/README.zh-CN.md)

AMCLab is an open-source Python toolkit for signal modulation classification research. It combines I/Q dataset loading, signal preprocessing and augmentation, built-in baseline models, and Keras-like training and evaluation APIs in a unified workflow.

The goal is to make it easier to develop new AMC models, reproduce experiments, and compare different architectures under consistent experimental settings, without rewriting the training pipeline.

## Features

- **Datasets:** loaders for RadioML 2016.10a/10b, RadioML 2018, RML22, HisarMod 2019, and synthetic I/Q data; lazy HDF5 loading for RadioML 2018.
- **Signal transforms:** composable preprocessing and augmentation, including power normalization and random phase rotation.
- **Baseline models:** 16 model families, including CNN, recurrent, residual, and hybrid architectures, with size variants for selected models.
- **Training:** Keras-like `Trainer.fit()` with checkpointing, early stopping, and support for resuming training.
- **Evaluation:** overall and high-SNR accuracy, per-SNR accuracy, macro-F1, Cohen's kappa, and confusion matrices.
- **Reproducibility:** persistent class- and SNR-stratified dataset splits, seed management, and automatic CUDA, MPS, or CPU selection.

## Installation

Python 3.9+ is specified in the package metadata. Choose a PyTorch build compatible with your Python version and hardware.

### Option 1: Install as a Python package (recommended)

Clone the repository and install it with the optional dataset dependencies:

```bash
git clone https://github.com/IvanDai/AMCLab.git
cd AMCLab
python -m pip install .
```

For development, use editable installation instead:

```bash
python -m pip install -e .
```

You can also install directly from GitHub (requires Git):

```bash
python -m pip install "amclab @ git+https://github.com/IvanDai/AMCLab.git"
```

### Option 2: Use directly in your own project

Copy the entire `src/amclab/` directory into the root of your Python project:

```text
your_project/
├── amclab/
└── train.py
```

Install the required third-party dependencies in your Python environment:

```bash
python -m pip install "torch>=2.0.0" "numpy>=1.24.0" "tqdm>=4.65.0" "h5py>=3.8.0" "scipy>=1.10.0"
```

Run your scripts from the project root. The local `amclab` package can be imported directly, without installing AMCLab itself or modifying `sys.path`. Retain the MIT license notice when redistributing the source.

Both installation methods use the same Python API:

```python
import amclab

print(amclab.__version__)
print(amclab.list_models(canonical_only=True))
```

## Quick Start

Train and evaluate a baseline model with synthetic data, without downloading a dataset:

```python
from amclab import Evaluator, Trainer, get_model, load_dataset, seed_everything

seed_everything(42)

train_ds, val_ds, test_ds = load_dataset(
    name="synthetic",
    split=(0.6, 0.2, 0.2),
    seed=42,
    split_file="./runs/quickstart/splits_seed42.npz",
)

model = get_model(
    "mcldnn",
    num_classes=train_ds.num_classes,
    seq_length=train_ds.seq_length,
)

trainer = Trainer(model, device="auto", save_dir="./runs/quickstart")
trainer.fit(train_ds, val_ds, epochs=3, batch_size=32)

evaluator = Evaluator(model, device="auto")
metrics = evaluator.evaluate(
    test_ds,
    checkpoint="./runs/quickstart/checkpoints/best.pt",
    save_dir="./runs/quickstart/eval",
)
```

For real datasets, set the dataset name and path accordingly. Reuse the same `split_file` across training, resumed training, and evaluation to keep the train/validation/test partitions consistent.

## Examples

Run the examples from the repository root after installing AMCLab:

| Example | Description |
| --- | --- |
| [01 — Single baseline](examples/01_single_baseline_train_resume/) | Train, resume, and evaluate a built-in model. |
| [02 — Custom model](examples/02_custom_model/) | Train and evaluate custom PyTorch model variants. |
| [03 — Batch benchmarking](examples/03_batch_train_and_eval/) | Train multiple baselines, evaluate saved checkpoints, and export comparison tables. |

```bash
python examples/01_single_baseline_train_resume/train_and_resume.py
python examples/02_custom_model/run_custom_model.py
python examples/03_batch_train_and_eval/01_batch_train.py
python examples/03_batch_train_and_eval/02_batch_eval_summary.py
```

The examples use synthetic data by default. To run on real datasets, edit `DATASET_NAME` and `DATASET_PATH` in the example scripts or the shared `config.py`. Experiment artifacts are saved under `runs/` by default and should remain local.

## Publications

<!-- Add related publications here. -->

## License & Support

AMCLab is released under the [MIT License](LICENSE).

If you find AMCLab useful in your research or projects, please consider giving the repository a ⭐ [Star](https://github.com/IvanDai/AMCLab). Your support is greatly appreciated!

## Repository Structure

```text
AMCLab/
├── src/amclab/       # Core Python package
│   ├── datasets/     # I/Q datasets and data splits
│   ├── transforms/   # Signal preprocessing and augmentation
│   ├── models/       # Baseline models
│   ├── engine/       # Training, evaluation, and metrics
│   └── utils/        # Runtime utilities
├── examples/         # Training and benchmarking examples
├── docs/             # Additional documentation
├── pyproject.toml    # Package configuration
└── requirements.txt # Dependencies
```
