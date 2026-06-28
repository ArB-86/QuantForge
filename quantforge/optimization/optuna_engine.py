import copy

import optuna

from quantforge.optimization.objectives import (
    portfolio_objective,
)

from quantforge.optimization.search_space import (
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

        metrics = self.runner(cfg)

        return portfolio_objective(
            metrics
        )

    def optimize(
        self,
        n_trials=50,
    ):

        study = optuna.create_study(
            direction="maximize",
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
        )

        return study