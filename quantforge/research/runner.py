from quantforge.pipeline.experiment_pipeline import ExperimentPipeline


class ExperimentRunner:

    def __init__(self):

        self.pipeline = ExperimentPipeline()

    def __call__(self, config):

        context = self.pipeline.run(config)

        return context.metrics
