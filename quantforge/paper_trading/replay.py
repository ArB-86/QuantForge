from typing import Iterable


class ReplayEngine:
    def __init__(self, engine):
        self.engine = engine

    def run(self, bars: Iterable[dict]):
        for bar in bars:
            ticker = bar["ticker"]
            price = float(bar["close"])

            self.engine.market.update(ticker, price)

            if "orders" in bar:
                for o in bar["orders"]:
                    self.engine.submit(
                        ticker=o["ticker"],
                        side=o["side"],
                        quantity=o["quantity"],
                        price=price,
                    )

            self.engine.snapshot(bar["date"])

        return self.engine.performance.history
