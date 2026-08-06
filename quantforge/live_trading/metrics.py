from dataclasses import dataclass


@dataclass
class LiveMetrics:
    trades: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    turnover: float = 0.0

    def update(self, fill):
        self.trades += 1

        if fill.side == "BUY":
            self.buy_orders += 1
        else:
            self.sell_orders += 1

        self.turnover += fill.quantity * fill.price
