from quantforge.storage.database.logger import ExperimentLogger


class LoggingStage:

    def __init__(self):

        self.logger = ExperimentLogger()

    def run(self, context):

        self.logger.log(
            context.config,
            context.metrics,
            context.metrics["Score"],
        )

        return context
