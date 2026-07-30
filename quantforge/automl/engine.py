from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import optuna

from quantforge.automl.objectives import portfolio_objective
from quantforge.automl.search_space import SEARCH_SPACE


class OptunaEngine:
    def __init__(self, base_config, runner):
        self.base_config = base_config
        self.runner = runner

    def _artifact_paths(self, cfg: Dict[str, Any], trial_id: int) -> Dict[str, str]:
        checkpoint = Path(cfg["checkpoint_file"])
        model = Path(cfg["model_file"])
        prediction = Path(cfg["prediction_file"])

        checkpoint = checkpoint.with_name(f"{checkpoint.stem}_trial_{trial_id}{checkpoint.suffix}")
        model = model.with_name(f"{model.stem}_trial_{trial_id}{model.suffix}")
        prediction = prediction.with_name(f"{prediction.stem}_trial_{trial_id}{prediction.suffix}")

        return {
            "checkpoint_file": str(checkpoint),
            "model_file": str(model),
            "prediction_file": str(prediction),
        }

    def _suggest_lightgbm(self, trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
        space = SEARCH_SPACE["lightgbm"]
        cfg["learning_rate"] = trial.suggest_float("learning_rate", *space["learning_rate"], log=True)
        cfg["num_leaves"] = trial.suggest_int("num_leaves", *space["num_leaves"])
        cfg["max_depth"] = trial.suggest_int("max_depth", *space["max_depth"])
        cfg["subsample"] = trial.suggest_float("subsample", *space["subsample"])
        cfg["colsample_bytree"] = trial.suggest_float("colsample_bytree", *space["colsample_bytree"])
        cfg["min_child_samples"] = trial.suggest_int("min_child_samples", *space["min_child_samples"])
        cfg["reg_alpha"] = trial.suggest_float("reg_alpha", *space["reg_alpha"])
        cfg["reg_lambda"] = trial.suggest_float("reg_lambda", *space["reg_lambda"])
        return cfg

    def _suggest_catboost(self, trial, cfg: Dict[str, Any]) -> Dict[str, Any]:
        space = SEARCH_SPACE["catboost"]
        cfg["iterations"] = trial.suggest_int("iterations", *space["iterations"])
        cfg["learning_rate"] = trial.suggest_float("learning_rate", *space["learning_rate"], log=True)
        cfg["depth"] = trial.suggest_int("depth", *space["depth"])
        cfg["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", *space["l2_leaf_reg"])
        cfg["bagging_temperature"] = trial.suggest_float("bagging_temperature", *space["bagging_temperature"])
        cfg["random_strength"] = trial.suggest_float("random_strength", *space["random_strength"])
        cfg["border_count"] = trial.suggest_int("border_count", *space["border_count"])
        cfg.pop("num_leaves", None)
        cfg.pop("min_child_samples", None)
        cfg.pop("colsample_bytree", None)
        cfg.pop("subsample_freq", None)
        cfg.pop("device", None)
        cfg["task_type"] = cfg.get("task_type", "GPU")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        cfg["devices"] = "0" if visible else cfg.get("devices", "0")
        cfg["verbose"] = bool(cfg.get("verbose", False))
        return cfg

    def objective(self, trial):
        cfg = copy.deepcopy(self.base_config)
        trial_id = trial.number

        cfg.update(self._artifact_paths(cfg, trial_id))

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
        score = portfolio_objective(metrics)

        if not metrics.get("Valid", True):
            penalty = 0.0
            sharpe = float(metrics.get("Sharpe", 0.0) or 0.0)
            max_dd = float(metrics.get("Max Drawdown", 0.0) or 0.0)
            win_rate = float(metrics.get("Win Rate", 0.0) or 0.0)

            if sharpe < 1.0:
                penalty += (1.0 - sharpe) * 5.0
            if max_dd < -0.40:
                penalty += abs(max_dd + 0.40) * 20.0
            if win_rate < 0.45:
                penalty += (0.45 - win_rate) * 10.0

            score -= penalty

        trial.set_user_attr("backend", backend)
        trial.set_user_attr("valid", bool(metrics.get("Valid", True)))
        trial.set_user_attr("checkpoint_file", cfg["checkpoint_file"])
        trial.set_user_attr("model_file", cfg["model_file"])
        trial.set_user_attr("prediction_file", cfg["prediction_file"])
        trial.set_user_attr("score", float(score))
        trial.set_user_attr("metrics", {k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool))})

        return score

    def optimize(
        self,
        n_trials=100,
        storage=None,
        study_name="QuantForge",
        load_if_exists=True,
    ):
        study_dir = Path("results/studies")
        study_dir.mkdir(parents=True, exist_ok=True)

        if storage is None:
            storage = f"sqlite:///{study_dir / 'quantforge.db'}"

        sampler = optuna.samplers.TPESampler(seed=int(self.base_config.get("seed", 42)))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=min(os.cpu_count() or 1, 8),
            show_progress_bar=True,
        )

        best_payload = {
            "study_name": study.study_name,
            "storage": storage,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "best_trial_user_attrs": dict(study.best_trial.user_attrs),
            "trials_summary": [
                {
                    "number": t.number,
                    "value": t.value,
                    "state": str(t.state),
                    "params": t.params,
                    "user_attrs": dict(t.user_attrs),
                }
                for t in study.trials
            ],
        }
        (study_dir / f"{study.study_name}_best.json").write_text(
            json.dumps(best_payload, indent=4, default=str)
        )

        print()
        print("=" * 80)
        print("BEST SCORE")
        print(study.best_value)
        print()
        print("BEST PARAMS")
        print(study.best_params)
        return study
