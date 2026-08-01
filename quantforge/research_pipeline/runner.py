from pathlib import Path
import time

import pandas as pd

from quantforge.analytics.trade_analyzer import (
    build_trade_log,
    compute_trade_statistics,
)
from quantforge.artifacts import ArtifactManager
from quantforge.backtest_engine.engine import backtest
from quantforge.benchmark.loader import load_benchmark
from quantforge.benchmark.metrics import compute_benchmark_metrics
from quantforge.core.config.config import Config
from quantforge.experiment.metrics import MetricsManager
from quantforge.experiment.registry import apply_experiment, get_experiment_type
from quantforge.research.runtime_dashboard import RuntimeDashboard
from quantforge.research_pipeline.context import ExperimentContext
from quantforge.research_pipeline.validation import ValidationManager
from quantforge.storage.database.logger import ExperimentLogger
from quantforge.automl.objectives import portfolio_objective
from quantforge.trainer.engine import train
from quantforge.walkforward import WalkForwardEngine


class ExperimentRunner:
    def __init__(self, config):
        if isinstance(config, str):
            config = Config(config).dict()
        self.config = config

    def _run_walkforward(self, context, dashboard):
        walk_cfg = dict(self.config)
        walk_cfg["name"] = self.config.get("name", "walkforward")
        engine = WalkForwardEngine(
            base_config=walk_cfg,
            runner=self.run,
            output_dir=str(Path("results") / "walkforward_optuna"),
        )
        dashboard.record("training_skipped", True)
        results = engine.run(
            n_trials_per_window=int(self.config.get("trials_per_window", 20)),
            storage=self.config.get("optuna_storage"),
            load_if_exists=True,
            max_windows=self.config.get("max_windows"),
        )
        context.metrics = {
            "WalkForwardWindows": len(results),
            "WalkForwardBestScore": max((r.get("best_value", float("-inf")) for r in results), default=float("-inf")),
        }
        context.artifacts["walkforward_summary"] = str(Path("results") / "walkforward_optuna" / "walkforward_summary.json")
        context.artifacts["walkforward_windows"] = str(len(results))
        return context

    def run(self, experiment="baseline"):
        self.config = apply_experiment(self.config, experiment)

        artifact_mgr = ArtifactManager(
            root="results/experiments",
            experiment=experiment,
        )

        self.config["model_file"] = str(artifact_mgr.model_file())
        self.config["prediction_file"] = str(artifact_mgr.prediction_file())
        self.config["checkpoint_file"] = str(artifact_mgr.checkpoint_file())
        artifact_mgr.save_config(self.config)

        dashboard = RuntimeDashboard(artifact_mgr.run_dir)
        dashboard.start_timer("total")

        context = ExperimentContext()
        context.experiment_id = f"EXP_{artifact_mgr.timestamp}"
        context.config = self.config
        context.status = "RUNNING"

        context.artifacts["experiment_dir"] = str(artifact_mgr.run_dir)
        context.artifacts["run_dir"] = str(artifact_mgr.run_dir)

        exp_type = get_experiment_type(experiment)
        if exp_type == "walkforward":
            context = self._run_walkforward(context, dashboard)
            ValidationManager(context).validate()
            logger = ExperimentLogger()
            score = portfolio_objective(context.metrics or {})
            context.metrics["Score"] = score
            logger.log(
                self.config,
                context.metrics,
                score,
                study=self.config.get("optuna_study"),
            )
            MetricsManager(context).save()
            artifact_mgr.save_metrics(context.metrics)
            context.status = "COMPLETED"
            dashboard.stop_timer("total")
            dashboard.save()
            return context

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
            skip_feature_importance = exp_type == "portfolio"
            dashboard.record("training_skipped", False)
            train(
                self.config,
                skip_feature_importance=skip_feature_importance,
                dashboard=dashboard,
            )

            if pred_path.exists():
                self.config["predictions_df"] = pd.read_parquet(pred_path)

        dashboard.start_timer("backtest")
        holdings, portfolio, metrics = backtest(self.config)
        artifact_mgr.save_holdings(holdings)
        trades = build_trade_log(
            holdings,
            return_column=self.config["target"],
        )

        trade_stats = compute_trade_statistics(trades)
        artifact_mgr.save_trade_log(trades)
        artifact_mgr.save_trade_stats(trade_stats)
        metrics.update(trade_stats)

        benchmark = load_benchmark(self.config)

        if benchmark is not None:
            benchmark_stats = compute_benchmark_metrics(portfolio, benchmark)
            metrics.update(benchmark_stats)
            artifact_mgr.save_benchmark(benchmark, benchmark_stats)

        dashboard.stop_timer("backtest")

        context.portfolio = portfolio
        context.holdings = holdings
        context.trades = trades
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
        logger.log(
            self.config,
            metrics,
            score,
            study=self.config.get("optuna_study"),
        )

        MetricsManager(context).save()
        artifact_mgr.save_metrics(metrics)
        portfolio.to_parquet(artifact_mgr.portfolio_file())

        context.status = "COMPLETED"
        dashboard.stop_timer("total")
        dashboard.save()

        return context