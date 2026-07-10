from typing import Callable, List


class Observer:
    def on_trade(self, engine, order):
        pass

    def on_snapshot(self, engine, snapshot):
        pass


class Observable:
    def __init__(self):
        self._observers: List[Observer] = []

    def register(self, observer: Observer):
        self._observers.append(observer)

    def notify_trade(self, engine, order):
        for obs in self._observers:
            obs.on_trade(engine, order)

    def notify_snapshot(self, engine, snapshot):
        for obs in self._observers:
            obs.on_snapshot(engine, snapshot)
