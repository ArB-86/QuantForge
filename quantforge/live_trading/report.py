from pathlib import Path
import json


class LiveTradingReport:

    def __init__(self, engine):
        self.engine = engine

    def generate(self, output_dir="results/live_trading"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics = self.engine.metrics

        report = {
            "trades": metrics.trades,
            "buy_orders": metrics.buy_orders,
            "sell_orders": metrics.sell_orders,
            "turnover": metrics.turnover,
            "orders": self.engine.router.orders(),
            "positions": self.engine.router.positions(),
            "holdings": self.engine.router.holdings(),
        }

        path = out / "report.json"

        with path.open("w") as f:
            json.dump(report, f, indent=2)

        return path
