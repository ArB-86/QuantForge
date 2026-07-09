from quantforge.pipeline.pipeline import Pipeline
from quantforge.pipeline.context import PipelineContext

from quantforge.pipeline.stages.config_stage import ConfigStage
from quantforge.pipeline.stages.dataset_stage import DatasetStage
from quantforge.pipeline.stages.training_stage import TrainingStage
from quantforge.pipeline.stages.backtest_stage import BacktestStage
from quantforge.pipeline.stages.validation_stage import ValidationStage
from quantforge.pipeline.stages.logging_stage import LoggingStage
from quantforge.pipeline.stages.artifact_stage import ArtifactStage


class ExperimentPipeline:

    def __init__(self):

        self.pipeline = (
            Pipeline()
            .add(ConfigStage())
            .add(DatasetStage())
            .add(TrainingStage())
            .add(BacktestStage())
            .add(ValidationStage())
            .add(LoggingStage())
            .add(ArtifactStage())
        )

    def run(self, config):

        context = PipelineContext(
            config=config
        )

        return self.pipeline.run(
            context
        )


def run(config):

    return (
        ExperimentPipeline()
        .run(config)
    )
