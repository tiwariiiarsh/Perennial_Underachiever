"""Central configuration: paths, blocking/model hyper-parameters, seeds."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", ROOT.parent / "student_resource" / "dataset"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", ROOT / "output"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", ROOT / "work"))      # cached intermediates
MODEL_DIR = Path(os.environ.get("ER_MODEL_DIR", ROOT / "models"))

SEED = 42
N_WORKERS = os.cpu_count() or 4

# ---- blocking ----
KEY_MAX_DF = 300         # pool-side keys shared by more records than this are ignored (too common)
TOP_K = 60              # candidates kept per Source-1 record
BLOCK_CHUNK = 20_000     # Source-1 records processed per blocking chunk

# ---- training ----
TRAIN_SAMPLE_S1 = 1_000_000  # Source-1 training entities used to fit the matcher
N_FOLDS = 5

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=600,
    verbose=-1,
    seed=SEED,
    num_threads=N_WORKERS,
)
