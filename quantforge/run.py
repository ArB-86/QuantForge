from quantforge.pipeline.experiment_pipeline import ExperimentPipeline


def run(config):

    context = ExperimentPipeline().run(config)

    return context.metrics
