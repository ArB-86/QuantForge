from typing import Dict
from quantforge.paper_trading.types import Position


class Portfolio:
    def __init__(self, initial_cash: float = 1_000_000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}

    def buy(self, ticker: str, qty: float, price: float):
        cost = qty * price
        if cost > self.cash:
            raise ValueError("Insufficient cash")

        self.cash -= cost

        if ticker in self.positions:
            p = self.positions[ticker]
            total_qty = p.quantity + qty
            p.avg_price = ((p.avg_price * p.quantity) + cost) / total_qty
            p.quantity = total_qty
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                avg_price=price,
            )

    def sell(self, ticker: str, qty: float, price: float):
        if ticker not in self.positions:
            raise ValueError("Position not found")

        p = self.positions[ticker]

        if qty > p.quantity:
            raise ValueError("Insufficient quantity")

        p.quantity -= qty
        self.cash += qty * price

        if p.quantity == 0:
            del self.positions[ticker]

    def equity(self, prices: Dict[str, float]) -> float:
        value = self.cash
        for t, p in self.positions.items():
            value += p.quantity * prices.get(t, p.avg_price)
        return value
