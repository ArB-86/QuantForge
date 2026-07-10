from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketTick:
    timestamp: datetime
    ticker: str
    ltp: float
    bid: float = None
    ask: float = None
    volume: float = 0


class MarketDataStore:

    def __init__(self):
        self._ticks = {}

    def update(self, tick: MarketTick):
        self._ticks[tick.ticker] = tick

    def get(self, ticker):
        return self._ticks.get(ticker)

    def snapshot(self):
        return dict(self._ticks)
