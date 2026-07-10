from quantforge.live_trading.execution_engine import LiveExecutionEngine
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

engine = LiveExecutionEngine()

fills = engine.execute([
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    ),
    LiveOrder(
        ticker="TCS.NS",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.MARKET,
    ),
])

assert len(fills) == 2

for f in fills:
    print(f)

print("PASS")
