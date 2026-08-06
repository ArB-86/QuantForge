from pathlib import Path

from quantforge.live_trading.execution_engine import LiveExecutionEngine
from quantforge.live_trading.order import LiveOrder, OrderSide, OrderType

engine = LiveExecutionEngine()

engine.execute([
    LiveOrder(
        ticker="RELIANCE.NS",
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.MARKET,
        price=1500,
    )
])

assert Path("results/live_trading/audit.jsonl").exists()

print("PASS")
