from quantforge.paper_trading.live_bridge import PaperTradingBridge
from quantforge.paper_trading.types import OrderSide

bridge = PaperTradingBridge()

bridge.on_predictions([
    {
        "ticker": "RELIANCE.NS",
        "side": OrderSide.BUY,
        "quantity": 100,
        "price": 1500,
    }
])

bridge.on_market(
    {"RELIANCE.NS": 1525},
    "2026-07-10",
)

print(bridge.engine.portfolio_value(
    bridge.engine.market.snapshot()
))
