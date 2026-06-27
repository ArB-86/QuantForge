"""
Centralized filesystem paths for QuantForge.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA = PROJECT_ROOT / "data"

TRAINING_DATA = DATA / "training"

CHECKPOINTS = DATA / "checkpoints"

MODELS = PROJECT_ROOT / "models"

RESULTS = PROJECT_ROOT / "results"

CONFIGS = PROJECT_ROOT / "configs"


MASTER_DATASET = (
    TRAINING_DATA /
    "master_v9.csv"
)


REGRESSION_CHECKPOINT = (
    CHECKPOINTS /
    "monthly_walkforward_regression_v15.csv"
)


ENSEMBLE_CHECKPOINT = (
    CHECKPOINTS /
    "monthly_walkforward_ensemble_v2.csv"
)


LIGHTGBM_MODEL = (
    MODELS /
    "monthly_lightgbm_regressor_v15.pkl"
)


CHECKPOINTS.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)
