from collections import defaultdict


class Observable:

    def __init__(self):
        self._observers = defaultdict(list)

    def subscribe(self, event, callback):
        self._observers[event].append(callback)

    def notify(self, event, payload):
        for callback in self._observers[event]:
            callback(payload)
