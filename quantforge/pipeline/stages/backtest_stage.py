from quantforge.engine.backtest import backtest


class BacktestStage:

    def run(self, context):

        print("=" * 80)
        print("BACKTEST STAGE")
        print("=" * 80)

        predictions, metrics = backtest(context.config)

        context.predictions = predictions
        context.metrics = metrics

        return context
