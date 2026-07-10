from typing import Dict


class Market:
    def __init__(self):
        self._prices: Dict[str, float] = {}

    def update(self, ticker: str, price: float):
        self._prices[ticker] = float(price)

    def update_many(self, prices: Dict[str, float]):
        self._prices.update({k: float(v) for k, v in prices.items()})

    def price(self, ticker: str) -> float:
        if ticker not in self._prices:
            raise KeyError(f"No market price for {ticker}")
        return self._prices[ticker]

    def snapshot(self):
        return dict(self._prices)
