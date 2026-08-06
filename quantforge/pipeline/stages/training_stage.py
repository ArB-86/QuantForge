from quantforge.trainer.engine import train


class TrainingStage:

    def run(self, context):

        print("=" * 80)
        print("TRAINING STAGE")
        print("=" * 80)

        context.model = train(context.config)

        return context
