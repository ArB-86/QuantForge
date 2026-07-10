from pathlib import Path
import json


class ReportGenerator:
    def __init__(self, engine):
        self.engine = engine

    def generate(self, output_dir="results/paper_trading"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report = {
            "cash": self.engine.cash(),
            "positions": {
                k: {
                    "quantity": v.quantity,
                    "avg_price": v.avg_price,
                }
                for k, v in self.engine.positions().items()
            },
            "performance": self.engine.performance.history,
            "ledger": self.engine.ledger.all(),
        }

        with open(out / "paper_trading_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(out / "paper_trading_report.json")
