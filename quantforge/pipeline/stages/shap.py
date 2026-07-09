from quantforge.pipeline.stage import Stage
from quantforge.shap.engine import shap

class SHAPStage(Stage):

    def run(self, context):

        print("="*80)
        print("SHAP")
        print("="*80)

        shap(context)

        return context
