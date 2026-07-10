from pathlib import Path

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
    )
])

path = engine.save_report()

assert Path(path).exists()

print(path)
print("PASS")
