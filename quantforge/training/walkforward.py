from quantforge.dataset.loader_builder import DatasetBuilder
from quantforge.training.model_manager import ModelManager
from quantforge.training.checkpoint_manager import CheckpointManager
from quantforge.training.monthly_loop import MonthlyLoop


class WalkForwardTrainer:

    def __init__(self, config):

        self.config = config

    def run(self):

        builder = DatasetBuilder(

            self.config["data_path"],

            self.config["features"],

            self.config["target"],

        )

        df = builder.prepare()

        checkpoint = CheckpointManager(

            self.config["checkpoint_file"],

            self.config["model_file"],

        )

        model_manager = ModelManager(

            self.config

        )

        loop = MonthlyLoop(

            df=df,

            features=builder.features,   # <-- use the resolved list, not config string

            target=self.config["target"],

            model_manager=model_manager,

            checkpoint_manager=checkpoint,

            purge_days=self.config.get(
                "purge_days",
                5
            ),

        )

        loop.run()

        return checkpoint