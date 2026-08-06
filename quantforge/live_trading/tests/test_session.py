from quantforge.live_trading.session import TradingSession
from datetime import time

s = TradingSession()

print(s.is_open(time(10,0)))
print(s.is_open(time(20,0)))

assert s.is_open(time(10,0))
assert not s.is_open(time(20,0))

print("PASS")
