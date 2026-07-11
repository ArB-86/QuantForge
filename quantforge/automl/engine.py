import copy
from pathlib import Path
import os

import optuna

from quantforge.automl.objectives import (
    portfolio_objective,
)

from quantforge.automl.search_space import (
    SEARCH_SPACE,
)


class OptunaEngine:

    def __init__(
        self,
        base_config,
        runner,
    ):

        self.base_config = base_config

        self.runner = runner

    def objective(
        self,
        trial,
    ):

        cfg = copy.deepcopy(
            self.base_config
        )

        trial_id = trial.number

        checkpoint = Path(
            cfg["checkpoint_file"]
        )

        model = Path(
            cfg["model_file"]
        )

        prediction = Path(
            cfg["prediction_file"]
        )

        checkpoint = checkpoint.with_name(
            checkpoint.stem + f"_trial_{trial_id}" + checkpoint.suffix
        )

        model = model.with_name(
            model.stem + f"_trial_{trial_id}" + model.suffix
        )

        prediction = prediction.with_name(
            prediction.stem + f"_trial_{trial_id}" + prediction.suffix
        )

        cfg["checkpoint_file"] = str(checkpoint)
        cfg["model_file"] = str(model)
        cfg["prediction_file"] = str(prediction)

        cfg["learning_rate"] = trial.suggest_float(
            "learning_rate",
            *SEARCH_SPACE["learning_rate"],
            log=True,
        )

        cfg["num_leaves"] = trial.suggest_int(
            "num_leaves",
            *SEARCH_SPACE["num_leaves"],
        )

        cfg["max_depth"] = trial.suggest_int(
            "max_depth",
            *SEARCH_SPACE["max_depth"],
        )

        cfg["subsample"] = trial.suggest_float(
            "subsample",
            *SEARCH_SPACE["subsample"],
        )

        cfg["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree",
            *SEARCH_SPACE["colsample_bytree"],
        )

        cfg["min_child_samples"] = trial.suggest_int(
            "min_child_samples",
            *SEARCH_SPACE["min_child_samples"],
        )

        cfg["reg_alpha"] = trial.suggest_float(
            "reg_alpha",
            *SEARCH_SPACE["reg_alpha"],
        )

        cfg["reg_lambda"] = trial.suggest_float(
            "reg_lambda",
            *SEARCH_SPACE["reg_lambda"],
        )

        print("=" * 80)
        print("TRIAL", trial.number)
        print("Checkpoint:", cfg["checkpoint_file"])
        print("Prediction:", cfg["prediction_file"])
        print("Model:", cfg["model_file"])
        print("=" * 80)

        metrics = self.runner(cfg)

        #
        # Soft penalties instead of hard rejection
        #

        score = portfolio_objective(metrics)

        if not metrics["Valid"]:

            penalty = 0.0

            if metrics["Sharpe"] < 1.0:
                penalty += (1.0 - metrics["Sharpe"]) * 5

            if metrics["Max Drawdown"] < -0.40:
                penalty += abs(metrics["Max Drawdown"] + 0.40) * 20

            if metrics["Win Rate"] < 0.45:
                penalty += (0.45 - metrics["Win Rate"]) * 10

            score -= penalty

        return score

    def optimize(
        self,
        n_trials=100,
    ):

        # Create directory for study storage
        study_dir = Path("results/studies")
        study_dir.mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name="QuantForge",
            storage=f"sqlite:///{study_dir/'quantforge.db'}",
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(

            self.objective,

            n_trials=n_trials,

            n_jobs=min(os.cpu_count(), 8),

            show_progress_bar=True,

        )

        print()

        print("=" * 80)

        print("BEST SCORE")

        print(study.best_value)

        print()

        print("BEST PARAMS")

        print(study.best_params)

        return study
