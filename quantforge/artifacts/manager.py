import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class ArtifactManager:
    """
    Centralized manager for all experiment artifacts.

    Creates a unique run directory and provides consistent paths for all outputs.
    """

    def __init__(
        self,
        root: str = "results/experiments",
        experiment: str = "baseline",
        timestamp: Optional[str] = None,
    ):
        """
        Initialize the artifact manager.

        Args:
            root: Base directory for experiments.
            experiment: Name of the experiment (used as a label).
            timestamp: Optional timestamp string; if not provided, generated automatically.
        """
        self.root = Path(root)
        self.experiment = experiment

        if timestamp is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            self.timestamp = f"{ts}_{uid}"
        else:
            self.timestamp = timestamp

        self._run_dir = self.root / f"EXP_{self.timestamp}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def metrics_file(self) -> Path:
        return self._run_dir / "metrics.json"

    def prediction_file(self) -> Path:
        return self._run_dir / "predictions.parquet"

    def portfolio_file(self) -> Path:
        return self._run_dir / "portfolio.parquet"

    def equity_file(self) -> Path:
        return self._run_dir / "equity.parquet"

    def feature_importance_file(self) -> Path:
        return self._run_dir / "feature_importance.csv"

    def metadata_file(self) -> Path:
        return self._run_dir / "metadata.json"

    def report_file(self) -> Path:
        return self._run_dir / "report.html"

    def config_file(self) -> Path:
        return self._run_dir / "config.json"

    def model_file(self) -> Path:
        return self._run_dir / "model.pkl"

    def checkpoint_file(self) -> Path:
        return self._run_dir / "checkpoint.csv"

    def save_config(self, config: dict) -> None:
        """Save the configuration to the run directory."""
        with open(self.config_file(), "w") as f:
            json.dump(config, f, indent=4, default=str)

    def save_metadata(self, metadata: dict) -> None:
        """Save metadata (e.g., experiment parameters)."""
        with open(self.metadata_file(), "w") as f:
            json.dump(metadata, f, indent=4, default=str)

    def save_metrics(self, metrics: dict) -> None:
        """Save metrics to the run directory."""
        with open(self.metrics_file(), "w") as f:
            json.dump(metrics, f, indent=4, default=str)

    def trade_log_file(self):
        return self._run_dir / "trades.parquet"

    def trade_stats_file(self):
        return self._run_dir / "trade_stats.json"

    def save_trade_log(self, trades):
        trades.to_parquet(
            self.trade_log_file(),
            index=False,
        )

    def save_trade_stats(self, stats):
        import json
        with open(
            self.trade_stats_file(),
            "w",
        ) as f:
            json.dump(
                stats,
                f,
                indent=4,
                default=str,
            )

    def holdings_file(self):
        return self._run_dir / "holdings.parquet"

    def save_holdings(self, df):
        df.to_parquet(
            self.holdings_file(),
            index=False,
        )

    def benchmark_file(self):
        return self._run_dir / "benchmark.parquet"

    def benchmark_stats_file(self):
        return self._run_dir / "benchmark_stats.json"

    def save_benchmark(self, df, stats):
        df.to_parquet(self.benchmark_file(), index=False)
        import json
        with open(self.benchmark_stats_file(), "w") as f:
            json.dump(stats, f, indent=4, default=str)
