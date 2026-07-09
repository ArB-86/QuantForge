from quantforge.pipeline.stage import Stage
from quantforge.prediction.engine import prediction

class PredictionStage(Stage):

    def run(self, context):

        print("="*80)
        print("PREDICTION")
        print("="*80)

        prediction(context)

        return context
