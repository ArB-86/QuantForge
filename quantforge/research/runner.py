from quantforge.trainer.engine import train
from quantforge.backtest_engine.engine import backtest

from quantforge.storage.database.logger import (
    ExperimentLogger,
)

from quantforge.automl.objectives import (
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

        print("\nTRIAL METRICS")
        for k, v in metrics.items():
            print(k, "=", v)
        print()

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

        score = portfolio_objective(metrics)

        if not valid:
            score -= 100

        metrics["Score"] = score

        self.logger.log(
            config,
            metrics,
            score,
        )

        if not valid:

            print("=" * 80)
            print("EXPERIMENT REJECTED")
            print(metrics["RejectReason"])
            print("=" * 80)

        return metrics