from copy import deepcopy
from pathlib import Path
import json

from quantforge.backtest.backtest_engine import BacktestEngine
from quantforge.experiments.manager import ExperimentManager


class ExperimentRunner:

    def __init__(self, config_path):

        self.manager = ExperimentManager(config_path)

        self.base_config = deepcopy(
            self.manager.config
        )

        self.predictions = (
            self.manager.load_predictions()
        )

    def run(self, **params):

        config = deepcopy(
            self.base_config
        )

        config.update(params)

        engine = BacktestEngine(config)

        # Prevent another CSV read
        engine.load_predictions = (
            lambda: self.predictions.copy()
        )

        _, metrics = engine.run()

        return metrics
