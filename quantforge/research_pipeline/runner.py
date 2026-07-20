from pathlib import Path
import pandas as pd
import time

from quantforge.research_pipeline.context import ExperimentContext
from quantforge.trainer.engine import train
from quantforge.backtest_engine.engine import backtest
from quantforge.artifacts import ArtifactManager
from quantforge.experiment.metrics import MetricsManager
from quantforge.research_pipeline.validation import ValidationManager
from quantforge.core.config.config import Config
from quantforge.storage.database.logger import ExperimentLogger
from quantforge.automl.objectives import portfolio_objective
from quantforge.experiment.registry import apply_experiment, get_experiment_type
from quantforge.research.runtime_dashboard import RuntimeDashboard


class ExperimentRunner:

    def __init__(self, config):
        if isinstance(config, str):
            config = Config(config).dict()
        self.config = config

    def run(
        self,
        experiment="baseline",
    ):

        # Apply experiment overrides
        self.config = apply_experiment(
            self.config,
            experiment,
        )

        # ---- Artifact Manager ----
        artifact_mgr = ArtifactManager(
            root="results/experiments",
            experiment=experiment,
        )
        artifact_mgr.save_config(self.config)

        # ---- Dashboard ----
        dashboard = RuntimeDashboard(artifact_mgr.run_dir)
        dashboard.start_timer("total")

        # ---- Create context ----
        context = ExperimentContext()
        context.experiment_id = f"EXP_{artifact_mgr.timestamp}"
        context.config = self.config
        context.status = "RUNNING"

        context.artifacts["experiment_dir"] = str(artifact_mgr.run_dir)
        context.artifacts["run_dir"] = str(artifact_mgr.run_dir)

        # ---- Determine experiment type ----
        exp_type = get_experiment_type(experiment)

        # ---- Prediction reuse ----
        pred_path = Path(self.config["prediction_file"])

        if exp_type == "portfolio" and pred_path.exists():
            print("=" * 80)
            print(f"[{experiment}] Using existing predictions from {pred_path}")
            print("=" * 80)
            self.config["predictions_df"] = pd.read_parquet(pred_path)
            dashboard.record("training_skipped", True)
        else:
            print("=" * 80)
            print(f"[{experiment}] Training model and generating predictions")
            print("=" * 80)
            skip_feature_importance = (exp_type == "portfolio")
            dashboard.record("training_skipped", False)
            checkpoint = train(self.config, skip_feature_importance=skip_feature_importance, dashboard=dashboard)

            if pred_path.exists():
                self.config["predictions_df"] = pd.read_parquet(pred_path)

        # ---- Backtest ----
        dashboard.start_timer("backtest")
        portfolio, metrics = backtest(self.config)
        dashboard.stop_timer("backtest")

        context.portfolio = portfolio
        context.metrics = metrics

        context.artifacts["checkpoint"] = self.config["checkpoint_file"]
        context.artifacts["model"] = self.config["model_file"]

        context.dataset_path = self.config["data_path"]
        context.target = self.config["target"]
        context.feature_names = self.config["features"]

        ValidationManager(context).validate()

        score = portfolio_objective(metrics)
        metrics["Score"] = score

        logger = ExperimentLogger()
        logger.log(self.config, metrics, score)

        MetricsManager(context).save()
        artifact_mgr.save_metrics(metrics)

        portfolio.to_parquet(artifact_mgr.portfolio_file())

        context.status = "COMPLETED"

        dashboard.stop_timer("total")
        dashboard.save()

        return context
        from quantforge.research.runtime_dashboard import RuntimeDashboard
        self.dashboard = RuntimeDashboard(artifact_mgr.run_dir)
        self.dashboard.start_timer("total")

        # ... (existing code) ...

        # When calling train, pass dashboard
        if exp_type == "portfolio" and pred_path.exists():
            ...
        else:
            checkpoint = train(self.config, skip_feature_importance=skip_feature_importance, dashboard=self.dashboard)

        # ... after backtest ...
        self.dashboard.stop_timer("total")
        self.dashboard.record("backtest_seconds", backtest_time)  # need to measure backtest time
        self.dashboard.save()
        from quantforge.research.runtime_dashboard import RuntimeDashboard
        self.dashboard = RuntimeDashboard(artifact_mgr.run_dir)
        self.dashboard.start_timer("total")

        # ... (existing code) ...

        # When calling train, pass dashboard
        if exp_type == "portfolio" and pred_path.exists():
            ...
        else:
            checkpoint = train(self.config, skip_feature_importance=skip_feature_importance, dashboard=self.dashboard)

        # ... after backtest ...
        self.dashboard.stop_timer("total")
        self.dashboard.record("backtest_seconds", backtest_time)  # need to measure backtest time
        self.dashboard.save()
