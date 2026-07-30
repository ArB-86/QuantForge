import argparse
import json

from quantforge.optimization.optuna_engine import OptunaEngine
from quantforge.research.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="QuantForge Optuna optimization")
    parser.add_argument(
        "--config",
        default="configs/lightgbm_optuna.json",
        help="Path to a QuantForge config file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///quantforge_optuna.db",
    )
    parser.add_argument(
        "--study-name",
        default="QuantForge",
        help="Optuna study name",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    runner = ExperimentRunner()
    engine = OptunaEngine(
        config,
        runner,
    )

    study = engine.optimize(
        n_trials=args.trials,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    print()
    print(study.best_value)
    print(study.best_params)


if __name__ == "__main__":
    main()
