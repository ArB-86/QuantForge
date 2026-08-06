from dataclasses import dataclass
from typing import List


@dataclass
class OrderBookEntry:
    price: float
    quantity: int


class OrderBook:

    def __init__(self):
        self.bids: List[OrderBookEntry] = []
        self.asks: List[OrderBookEntry] = []

    def update(self, bids, asks):
        self.bids = [
            OrderBookEntry(*x) for x in bids
        ]
        self.asks = [
            OrderBookEntry(*x) for x in asks
        ]

    @property
    def best_bid(self):
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self):
        return self.asks[0] if self.asks else None

    @property
    def spread(self):
        if not self.best_bid or not self.best_ask:
            return None
        return self.best_ask.price - self.best_bid.price
