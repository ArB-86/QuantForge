from quantforge.training.checkpoint_manager import CheckpointManager
from quantforge.training.monthly_loop import MonthlyLoop
from quantforge.models.factory import build


class TrainingStage:

    def run(self, context):

        checkpoint = CheckpointManager(

            context.config["checkpoint_file"],

            context.config["model_file"],

        )

        model = build(context.config)

        loop = MonthlyLoop(

            df=context.dataset,

            features=context.features,

            target=context.config["target"],

            model_manager=model,

            checkpoint_manager=checkpoint,

            purge_days=context.config.get(
                "purge_days",
                5,
            ),

        )

        loop.run()

        context.model = model

        return context
