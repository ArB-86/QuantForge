from dataclasses import dataclass, asdict
from typing import List


@dataclass
class LedgerEntry:
    timestamp: str
    ticker: str
    side: str
    quantity: float
    price: float
    value: float


class Ledger:
    def __init__(self):
        self.entries: List[LedgerEntry] = []

    def record(self, timestamp, ticker, side, quantity, price):
        self.entries.append(
            LedgerEntry(
                timestamp=timestamp,
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=price,
                value=quantity * price,
            )
        )

    def all(self):
        return [asdict(x) for x in self.entries]

    def __len__(self):
        return len(self.entries)
