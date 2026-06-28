from quantforge.pipeline.stage import Stage
from quantforge.engine.topn import topn

class TopNStage(Stage):

    def run(self, context):

        print("="*80)
        print("TOP-N")
        print("="*80)

        topn(context)

        return context
