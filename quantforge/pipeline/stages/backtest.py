from quantforge.pipeline.stage import Stage
from quantforge.backtest_engine.engine import backtest

class BacktestStage(Stage):

    def run(self, context):

        print("="*80)
        print("BACKTEST")
        print("="*80)

        metrics = backtest(context)

        context["metrics"] = metrics

        return context
