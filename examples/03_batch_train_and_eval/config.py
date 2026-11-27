"""Shared dataset, split, and model list configuration for Example 03."""

from pathlib import Path
from amclab.models import BENCHMARK_MODELS

# ==============================================================================
# DATASET LOCATION (Edit these two lines to run on your real dataset)
# ==============================================================================
DATASET_NAME = "synthetic"  # Options: "rml2016.10a", "rml2016.10b", "rml22", "rml2018", "hisarmod2019", "synthetic"
DATASET_PATH = None         # e.g., "/data/AMC/RML2016.10a_dict.pkl" or "/data/AMC/GOLD_XYZ_OSC.0001_1024.hdf5"
# ==============================================================================

OUTPUT_ROOT = Path("./runs/example03_benchmark")
SPLIT_FILE = OUTPUT_ROOT / "shared_splits_seed42.npz"

# Train and evaluate ALL 16 built-in baseline models by default.
# You can replace this with a subset, e.g. ["cnn1", "resnet", "mcldnn", "iqformer"].
MODELS_TO_RUN = list(BENCHMARK_MODELS)

SEED = 42
EPOCHS = 3          # Increase to 50-100 for real datasets
BATCH_SIZE = 32     # Increase to 128-256 for real datasets
EVAL_BATCH_SIZE = 64
LR = 1e-3
