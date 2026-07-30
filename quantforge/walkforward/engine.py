from __future__ import annotations

from typing import Any, Dict, List, Optional

from quantforge.dataset.builder import DatasetBuilder
from quantforge.features.registry import get_features
from quantforge.research.feature_importance import feature_importance
from quantforge.training.model_manager import ModelManager
from quantforge.walkforward.checkpoint import CheckpointManager
from quantforge.walkforward.monthly import MonthlyLoop
from quantforge.walkforward.optuna_walkforward import (
    WalkForwardStudyManager,
    WalkForwardWindow,
)


class WalkForwardTrainer:
    def __init__(self, config, skip_feature_importance: bool = False, dashboard=None):
        self.config = config
        self.skip_feature_importance = skip_feature_importance
        self.dashboard = dashboard

    def run(self):
        feature_set = self.config.get("feature_set", "v4")
        features = get_features(feature_set)

        builder = DatasetBuilder(
            self.config["data_path"],
            features,
            self.config["target"],
        )

        if self.dashboard:
            self.dashboard.start_timer("dataset_build")
        df = builder.prepare()
        if self.dashboard:
            self.dashboard.stop_timer("dataset_build")

        checkpoint = CheckpointManager(
            self.config["checkpoint_file"],
            self.config["model_file"],
        )

        model_manager = ModelManager(self.config)

        loop = MonthlyLoop(
            df=df,
            features=builder.features,
            target=self.config["target"],
            model_manager=model_manager,
            checkpoint_manager=checkpoint,
            prediction_file=self.config["prediction_file"],
            purge_days=self.config.get("purge_days", 5),
            dashboard=self.dashboard,
        )

        loop.run()

        if not self.skip_feature_importance:
            if self.dashboard:
                self.dashboard.start_timer("feature_importance")
            feature_importance(
                self.config["model_file"],
                builder.features,
            )
            if self.dashboard:
                self.dashboard.stop_timer("feature_importance")

        return checkpoint


class WalkForwardEngine:
    """Compatibility wrapper around the walk-forward Optuna manager."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        runner=None,
        output_dir: str = "results/walkforward_optuna",
    ):
        self.manager = WalkForwardStudyManager(
            base_config=base_config,
            runner=runner,
            output_dir=output_dir,
        )

    def run(
        self,
        n_trials_per_window: int = 50,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        max_windows: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.manager.run(
            n_trials_per_window=n_trials_per_window,
            storage=storage,
            load_if_exists=load_if_exists,
            max_windows=max_windows,
        )

    def build_windows(self, *args, **kwargs):
        return self.manager.build_windows(*args, **kwargs)

    def optimize_window(self, *args, **kwargs):
        return self.manager.optimize_window(*args, **kwargs)
