from quantforge.paper_trading.engine import PaperTradingEngine
from quantforge.paper_trading.data.replay_feed import ReplayFeed

engine = PaperTradingEngine()

feed = ReplayFeed.from_csv("data/sample_prices.csv")

for bar in feed:
    engine.mark(bar["prices"])
    engine.snapshot(bar["date"])

engine.save_report()

print(engine.performance.latest())
