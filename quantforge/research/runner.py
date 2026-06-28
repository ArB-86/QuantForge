from quantforge.engine.trainer import train
from quantforge.engine.backtest import backtest


class ExperimentRunner:

    def __call__(
        self,
        config,
    ):

        #
        # 1. Train
        #

        train(config)

        #
        # 2. Backtest
        #

        _, metrics = backtest(config)

        return metrics