from quantforge.artifacts import ArtifactManager


class ArtifactStage:

    def __init__(self):

        self.manager = ArtifactManager()

    def run(self, context):

        context.artifacts = self.manager.save(
            config=context.config,
            metrics=context.metrics,
            predictions=context.predictions,
            model=context.model,
        )

        context.experiment_id = context.artifacts["experiment_id"]

        return context
