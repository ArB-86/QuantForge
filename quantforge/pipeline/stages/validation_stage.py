from quantforge.research.validator import ExperimentValidator
from quantforge.optimization_engine.objectives import portfolio_objective


class ValidationStage:

    def __init__(self):

        self.validator = ExperimentValidator()

    def run(self, context):

        valid, reasons = self.validator.validate(
            context.metrics
        )

        context.metrics["Valid"] = valid
        context.metrics["RejectReason"] = "; ".join(reasons)

        score = portfolio_objective(
            context.metrics
        )

        if not valid:
            score -= 100

        context.metrics["Score"] = score

        return context
