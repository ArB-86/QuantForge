from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class LiveOrder:
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: float = None
