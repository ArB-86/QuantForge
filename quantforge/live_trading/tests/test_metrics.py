from quantforge.live_trading.execution_engine import LiveExecutionEngine
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

engine = LiveExecutionEngine()

engine.execute([
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        price=1500,
    ),
    LiveOrder(
        ticker="TCS.NS",
        side=OrderSide.SELL,
        quantity=5,
        order_type=OrderType.MARKET,
        price=3500,
    ),
])

m = engine.metrics

print(m)

assert m.trades == 2
assert m.buy_orders == 1
assert m.sell_orders == 1

print("PASS")
