from typing import Callable

from quantforge.live_trading.event_bus import EventBus
from quantforge.live_trading.market_data import (
    MarketDataStore,
    MarketTick,
)


class MarketStream:

    def __init__(self):
        self.store = MarketDataStore()
        self.callbacks = []
        self.bus = EventBus()

    def subscribe(self, callback: Callable):
        self.callbacks.append(callback)

    def publish(self, tick: MarketTick):
        self.store.update(tick)

        for cb in self.callbacks:
            cb(tick)

        self.bus.publish('tick', tick)

    def snapshot(self):
        return self.store.snapshot()


    def on_tick(self, handler):
        self.bus.subscribe("tick", handler)
