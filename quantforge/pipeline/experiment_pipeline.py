from quantforge.research_pipeline.pipeline import Pipeline
from quantforge.pipeline.context import PipelineContext

from quantforge.research_pipeline.stages.config_stage import ConfigStage
from quantforge.research_pipeline.stages.dataset_stage import DatasetStage
from quantforge.research_pipeline.stages.training_stage import TrainingStage
from quantforge.research_pipeline.stages.backtest_stage import BacktestStage
from quantforge.research_pipeline.stages.validation_stage import ValidationStage
from quantforge.research_pipeline.stages.logging_stage import LoggingStage
from quantforge.research_pipeline.stages.artifact_stage import ArtifactStage


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
