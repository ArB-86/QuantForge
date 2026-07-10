from quantforge.paper_trading.engine import PaperTradingEngine


class PaperTradingBridge:
    def __init__(self, engine=None):
        self.engine = engine or PaperTradingEngine()

    def on_predictions(self, predictions):
        for p in predictions:
            self.engine.submit(
                ticker=p["ticker"],
                side=p["side"],
                quantity=p["quantity"],
                price=p["price"],
            )

    def on_market(self, prices, timestamp):
        self.engine.mark(prices)
        self.engine.snapshot(timestamp)

    def report(self):
        self.engine.save_report()
