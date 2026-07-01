from quantforge.engine.trainer import train
from quantforge.engine.backtest import backtest

from quantforge.database.logger import (
    ExperimentLogger,
)

from quantforge.optimization.objectives import (
    portfolio_objective,
)

from quantforge.research.validator import (
    ExperimentValidator,
)


class ExperimentRunner:

    def __init__(self):

        self.logger = ExperimentLogger()

        self.validator = ExperimentValidator()

    def __call__(self, config):

        #
        # Train
        #

        train(config)

        #
        # Backtest
        #

        _, metrics = backtest(config)

        #
        # Validate
        #

        valid, reasons = self.validator.validate(
            metrics
        )

        metrics["Valid"] = valid
        metrics["RejectReason"] = "; ".join(
            reasons
        )

        if not valid:

            print("=" * 80)
            print("EXPERIMENT REJECTED")
            print(metrics["RejectReason"])
            print("=" * 80)

            return metrics

        #
        # Score
        #

        score = portfolio_objective(
            metrics
        )

        #
        # Log
        #

        self.logger.log(

            config,

            metrics,

            score,

        )

        metrics["Score"] = score

        return metrics