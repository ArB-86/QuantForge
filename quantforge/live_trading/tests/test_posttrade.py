from quantforge.live_trading.execution_engine import LiveExecutionEngine
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

engine = LiveExecutionEngine()

fills = engine.execute([
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.MARKET,
        price=1500,
    )
])

print(fills[0])

assert fills

print("PASS")
