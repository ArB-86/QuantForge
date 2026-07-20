from pathlib import Path

from quantforge.core.config.config import Config
from quantforge.experiment.registry import apply_experiment
from quantforge.validation.benchmark import BenchmarkEngine


def benchmark(config_path, experiment="baseline"):
    config = Config(config_path).dict()
    config = apply_experiment(config, experiment)
    engine = BenchmarkEngine(
        config["prediction_file"],
        config["holding_days"],
    )
    engine.report()
