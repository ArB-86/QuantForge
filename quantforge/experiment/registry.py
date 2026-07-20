from copy import deepcopy
from pathlib import Path

EXPERIMENTS = {
    "baseline": {
        "feature_set": "v4",
        "prediction_file": "../data/checkpoints/baseline_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/baseline_progress.csv",
        "model_file": "../models/baseline.pkl",
        "type": "train",
    },

    "v5": {
        "feature_set": "v5",
        "prediction_file": "../data/checkpoints/v5_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/v5_progress.csv",
        "model_file": "../models/v5.pkl",
        "type": "train",
    },

    "hold10": {
        "holding_days": 10,
        "prediction_file": "../data/checkpoints/hold10_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/hold10_progress.csv",
        "model_file": "../models/hold10.pkl",
        "type": "portfolio",
    },

    "hold20": {
        "holding_days": 20,
        "prediction_file": "../data/checkpoints/hold20_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/hold20_progress.csv",
        "model_file": "../models/hold20.pkl",
        "type": "portfolio",
    },

    "top10": {
        "top_n": 10,
        "prediction_file": "../data/checkpoints/top10_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/top10_progress.csv",
        "model_file": "../models/top10.pkl",
        "type": "portfolio",
    },

    "top20": {
        "top_n": 20,
        "prediction_file": "../data/checkpoints/top20_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/top20_progress.csv",
        "model_file": "../models/top20.pkl",
        "type": "portfolio",
    },

    "equal_weight": {
        "portfolio": "equal_weight",
        "prediction_file": "../data/checkpoints/equal_weight_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/equal_weight_progress.csv",
        "model_file": "../models/equal_weight.pkl",
        "type": "portfolio",
    },

    "risk_parity": {
        "portfolio": "risk_parity",
        "prediction_file": "../data/checkpoints/risk_parity_predictions.parquet",
        "checkpoint_file": "../data/checkpoints/risk_parity_progress.csv",
        "model_file": "../models/risk_parity.pkl",
        "type": "portfolio",
    },
}


def _suffix(path, name):
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{name}{p.suffix}"))


def apply_experiment(config, experiment):
    config = deepcopy(config)

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Override with experiment-specific paths
    for key, value in EXPERIMENTS[experiment].items():
        config[key] = value

    return config


def get_experiment_type(experiment):
    return EXPERIMENTS.get(experiment, {}).get("type", "train")
