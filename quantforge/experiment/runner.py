from quantforge.experiment.context import ExperimentContext
from quantforge.engine.trainer import train


class ExperimentRunner:

    def __init__(self, config):
        self.config = config

    def run(self):

        context = ExperimentContext()

        context.config = self.config
        context.status = "RUNNING"

        checkpoint = train(self.config)

        context.status = "COMPLETED"

        # Store outputs here as we migrate the codebase
        context.artifacts["checkpoint"] = checkpoint

        return context
