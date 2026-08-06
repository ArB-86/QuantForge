from datetime import datetime

from quantforge.live_trading.market_hours import MarketHours

assert MarketHours.is_open(datetime(2026,7,13,10,0))
assert not MarketHours.is_open(datetime(2026,7,12,10,0))
assert not MarketHours.is_open(datetime(2026,7,13,20,0))

print("PASS")
