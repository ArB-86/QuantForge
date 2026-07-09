import copy
from pathlib import Path
import os

import optuna

from quantforge.optimization_engine.objectives import (
    portfolio_objective,
)

from quantforge.optimization_engine.search_space import (
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
        # Soft penalty instead of hard rejection
        #

        if not metrics["Valid"]:

            return (

                metrics.get(
                    "Sharpe",
                    0,
                )

                - 100

                - abs(
                    metrics.get(
                        "Max Drawdown",
                        1,
                    )
                )

            )

        return portfolio_objective(
            metrics
        )

    def optimize(
        self,
        n_trials=100,
    ):

        study = optuna.create_study(

            study_name="QuantForge",

            direction="maximize",

        )

        study.optimize(

            self.objective,

            n_trials=n_trials,

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