from quantforge.experiment.context import ExperimentContext
from quantforge.engine.trainer import train
from quantforge.backtesting.engine import backtest
from quantforge.experiment.artifact_manager import ArtifactManager
from quantforge.experiment.metrics import MetricsManager
from quantforge.experiment.validation import ValidationManager
from quantforge.experiment.manager import ExperimentManager
from quantforge.config.config import Config


class ExperimentRunner:

    def __init__(self, config):
        # Convert config to dictionary if it's a string path
        if isinstance(config, str):
            config = Config(config).dict()

        self.config = config

    def run(self):

        manager = ExperimentManager(
            self.config.get(
                "config_file",
                "configs/lightgbm_regressor.json",
            )
        )

        experiment_id, experiment_dir = manager.create()

        context = ExperimentContext()

        context.experiment_id = experiment_id
        context.config = self.config
        context.status = "RUNNING"

        context.artifacts["experiment_dir"] = str(experiment_dir)

        # Train the model
        checkpoint = train(self.config)

        # Run backtest on the trained model
        portfolio, metrics = backtest(self.config)

        # Store backtest results in context
        context.portfolio = portfolio
        context.metrics = metrics

        # Store serializable metadata in the context
        context.artifacts["checkpoint"] = self.config["checkpoint_file"]
        context.artifacts["model"] = self.config["model_file"]

        context.dataset_path = self.config["data_path"]
        context.target = self.config["target"]
        context.feature_names = self.config["features"]

        ValidationManager(context).validate()
        ValidationManager(context).validate()
        MetricsManager(context).save()
        ArtifactManager(context).save()

        context.status = "COMPLETED"

        return context
