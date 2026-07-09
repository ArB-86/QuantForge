from quantforge.research_pipeline.stage import Stage
from quantforge.analysis_engine.prediction import prediction

class PredictionStage(Stage):

    def run(self, context):

        print("="*80)
        print("PREDICTION")
        print("="*80)

        prediction(context)

        return context
