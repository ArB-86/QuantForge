from dataclasses import dataclass
from quantforge.paper_trading.types import OrderSide, OrderStatus

@dataclass
class Order:
    ticker: str
    side: OrderSide
    quantity: float
    price: float
    status: OrderStatus = OrderStatus.NEW
