from quantforge.live_trading.engine import LiveTradingEngine
from quantforge.live_trading.order import (
    LiveOrder,
    OrderSide,
    OrderType,
)

engine = LiveTradingEngine()

engine.login()

engine.submit(
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )
)

print(engine.orders())
print(engine.funds())
