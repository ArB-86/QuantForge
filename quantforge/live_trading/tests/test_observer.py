from quantforge.live_trading.execution_engine import LiveExecutionEngine
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

events = []

engine = LiveExecutionEngine()

engine.subscribe(
    "fill",
    lambda fill: events.append(fill.ticker),
)

engine.execute([
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        price=1500,
    )
])

assert events == ["RELIANCE.NS"]

print("PASS")
