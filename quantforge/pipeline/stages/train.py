from quantforge.pipeline.stage import Stage
from quantforge.engine.trainer import train

class TrainStage(Stage):

    def run(self, context):

        print("="*80)
        print("TRAIN")
        print("="*80)

        model = train(context["config"])

        context["model"] = model

        return context
