from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeFill:
    timestamp: str
    ticker: str
    side: str
    quantity: float
    price: float
    order_id: str


class PostTradeProcessor:

    def __init__(self):
        self.fills = []

    def record(self, order, broker_response):

        fill = TradeFill(
            timestamp=datetime.utcnow().isoformat(),
            ticker=order.ticker,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price or 0.0,
            order_id=str(broker_response),
        )

        self.fills.append(fill)

        return fill
