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
from quantforge.analytics.trade_analyzer import (
    build_trade_log,
    compute_trade_statistics,
)
from quantforge.benchmark.loader import load_benchmark
from quantforge.benchmark.metrics import compute_benchmark_metrics
from quantforge.analytics.attribution import compute_attribution
from quantforge.attribution.engine import compute_attribution


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

        # ---- Update config with artifact paths BEFORE saving ----
        self.config["model_file"] = str(artifact_mgr.model_file())
        self.config["prediction_file"] = str(artifact_mgr.prediction_file())
        self.config["checkpoint_file"] = str(artifact_mgr.checkpoint_file())

        # ---- Now save the config (with correct paths) ----
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
        holdings, portfolio, metrics = backtest(self.config)
        artifact_mgr.save_holdings(holdings)
        trades = build_trade_log(
            holdings,
            return_column=self.config["target"],
        )

        trade_stats = compute_trade_statistics(
            trades,
        )

        artifact_mgr.save_trade_log(
            trades,
        )

        artifact_mgr.save_trade_stats(
            trade_stats,
        )

        metrics.update(
            trade_stats
        )
        # ---- Trade‑based Attribution ----
        attribution = compute_attribution(trades)
        artifact_mgr.save_attribution(attribution)

        if attribution.get("Top Contributors"):
            metrics["Top Contributor"] = next(iter(attribution["Top Contributors"]))
        if attribution.get("Worst Contributors"):
            metrics["Worst Contributor"] = next(iter(attribution["Worst Contributors"]))

        # ---- Benchmark ----
        benchmark = load_benchmark(self.config)
        benchmark_stats = compute_benchmark_metrics(portfolio, benchmark, config=self.config, strategy_cagr=metrics.get('CAGR'), holding_days=self.config.get('holding_days', 20))
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
        logger.log(self.config, metrics, score)

        MetricsManager(context).save()
        artifact_mgr.save_metrics(metrics)

        portfolio.to_parquet(artifact_mgr.portfolio_file())

        context.status = "COMPLETED"

        dashboard.stop_timer("total")
        dashboard.save()

        return context