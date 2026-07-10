from typing import Callable


class StrategyRunner:
    def __init__(self, engine, strategy: Callable):
        self.engine = engine
        self.strategy = strategy

    def step(self, market_snapshot):
        orders = self.strategy(
            market_snapshot,
            self.engine.positions(),
            self.engine.cash(),
        )

        for o in orders:
            self.engine.submit(
                ticker=o["ticker"],
                side=o["side"],
                quantity=o["quantity"],
                price=market_snapshot[o["ticker"]],
            )

        self.engine.mark(market_snapshot)
        self.engine.snapshot("STEP")

    def run(self, stream):
        for snapshot in stream:
            self.step(snapshot)
