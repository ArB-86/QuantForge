from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import optuna

from quantforge.automl.objectives import portfolio_objective
from quantforge.automl.search_space import SEARCH_SPACE


class OptunaEngine:
    def __init__(self, base_config, runner):
        self.base_config = base_config
        self.runner = runner

    def _trial_artifact_paths(self, cfg: Dict[str, Any], trial_id: int) -> Dict[str, str]:
        checkpoint = Path(cfg["checkpoint_file"])
        model = Path(cfg["model_file"])
        prediction = Path(cfg["prediction_file"])

        checkpoint = checkpoint.with_name(
            checkpoint.stem + f"_trial_{trial_id}" + checkpoint.suffix
        )
        model = model.with_name(model.stem + f"_trial_{trial_id}" + model.suffix)
        prediction = prediction.with_name(
            prediction.stem + f"_trial_{trial_id}" + prediction.suffix
        )

        return {
            "checkpoint_file": str(checkpoint),
            "model_file": str(model),
            "prediction_file": str(prediction),
        }

    def _suggest_lightgbm(self, trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
        space = SEARCH_SPACE["lightgbm"]
        cfg["learning_rate"] = trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        )
        cfg["num_leaves"] = trial.suggest_int("num_leaves", *space["num_leaves"])
        cfg["max_depth"] = trial.suggest_int("max_depth", *space["max_depth"])
        cfg["subsample"] = trial.suggest_float("subsample", *space["subsample"])
        cfg["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree", *space["colsample_bytree"]
        )
        cfg["min_child_samples"] = trial.suggest_int(
            "min_child_samples", *space["min_child_samples"]
        )
        cfg["reg_alpha"] = trial.suggest_float("reg_alpha", *space["reg_alpha"])
        cfg["reg_lambda"] = trial.suggest_float("reg_lambda", *space["reg_lambda"])
        return cfg

    def _suggest_catboost(self, trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
        space = SEARCH_SPACE["catboost"]
        cfg["iterations"] = trial.suggest_int("iterations", *space["iterations"])
        cfg["learning_rate"] = trial.suggest_float(
            "learning_rate", *space["learning_rate"], log=True
        )
        cfg["max_depth"] = trial.suggest_int("depth", *space["depth"])
        cfg["reg_lambda"] = trial.suggest_float("l2_leaf_reg", *space["l2_leaf_reg"])
        cfg["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", *space["bagging_temperature"]
        )
        cfg["random_strength"] = trial.suggest_float(
            "random_strength", *space["random_strength"]
        )
        cfg["border_count"] = trial.suggest_int("border_count", *space["border_count"])
        cfg["task_type"] = cfg.get("task_type", "GPU")
        return cfg

    def objective(self, trial):
        cfg = copy.deepcopy(self.base_config)
        trial_id = trial.number

        cfg.update(self._trial_artifact_paths(cfg, trial_id))

        backend = str(cfg.get("model", "lightgbm")).lower().strip()
        if backend == "catboost":
            cfg = self._suggest_catboost(trial, cfg)
        else:
            cfg = self._suggest_lightgbm(trial, cfg)

        print("=" * 80)
        print("TRIAL", trial.number)
        print("Backend:", backend)
        print("Checkpoint:", cfg["checkpoint_file"])
        print("Prediction:", cfg["prediction_file"])
        print("Model:", cfg["model_file"])
        print("=" * 80)

        metrics = self.runner(cfg)

        if not metrics["Valid"]:
            return (
                metrics.get("Sharpe", 0)
                - 100
                - abs(metrics.get("Max Drawdown", 1))
            )

        return portfolio_objective(metrics)

    def optimize(self, n_trials=100, storage=None, study_name="QuantForge", load_if_exists=True):
        sampler = optuna.samplers.TPESampler(seed=self.base_config.get("seed", 42))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists,
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
