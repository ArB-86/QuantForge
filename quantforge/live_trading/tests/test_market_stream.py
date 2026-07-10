from datetime import datetime

from quantforge.live_trading.market_stream import MarketStream
from quantforge.live_trading.market_data import MarketTick

stream = MarketStream()

received = []

stream.subscribe(
    lambda tick: received.append(tick.ticker)
)

stream.publish(
    MarketTick(
        timestamp=datetime.now(),
        ticker="RELIANCE.NS",
        ltp=1525.5,
    )
)

assert received == ["RELIANCE.NS"]

assert "RELIANCE.NS" in stream.snapshot()

print("PASS")
