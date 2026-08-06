from datetime import datetime

from quantforge.live_trading.market_data import (
    MarketDataStore,
    MarketTick,
)

store = MarketDataStore()

store.update(
    MarketTick(
        timestamp=datetime.now(),
        ticker="RELIANCE.NS",
        ltp=1525.50,
        bid=1525.25,
        ask=1525.75,
        volume=125000,
    )
)

tick = store.get("RELIANCE.NS")

assert tick.ltp == 1525.50
assert tick.bid == 1525.25
assert tick.ask == 1525.75

print(store.snapshot())
print("PASS")
