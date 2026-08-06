from quantforge.paper_trading.engine import PaperTradingEngine
from quantforge.paper_trading.types import OrderSide
from quantforge.paper_trading.strategy_runner import StrategyRunner


def strategy(snapshot, positions, cash):
    if positions:
        return []

    ticker = next(iter(snapshot.keys()))

    return [{
        "ticker": ticker,
        "side": OrderSide.BUY,
        "quantity": 100,
    }]


engine = PaperTradingEngine(1_000_000)

runner = StrategyRunner(engine, strategy)

runner.run([
    {"RELIANCE.NS": 1500},
    {"RELIANCE.NS": 1510},
    {"RELIANCE.NS": 1525},
    {"RELIANCE.NS": 1530},
])

print(engine.cash())
print(engine.portfolio_value(engine.market.snapshot()))

engine.save_report()
