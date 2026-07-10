from datetime import datetime

from quantforge.live_trading.market_stream import MarketStream
from quantforge.live_trading.market_data import MarketTick

events = []

stream = MarketStream()

stream.on_tick(lambda tick: events.append(tick.ticker))

stream.publish(
    MarketTick(
        timestamp=datetime.now(),
        ticker="INFY.NS",
        ltp=1600,
    )
)

assert events == ["INFY.NS"]

print("PASS")
