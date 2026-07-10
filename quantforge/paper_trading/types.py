from dataclasses import dataclass
from enum import Enum, auto

class OrderSide(Enum):
    BUY = auto()
    SELL = auto()

class OrderStatus(Enum):
    NEW = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()

@dataclass
class Position:
    ticker: str
    quantity: float
    avg_price: float

@dataclass
class Cash:
    balance: float
