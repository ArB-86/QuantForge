from quantforge.live_trading.broker import Broker


class PaperBroker(Broker):

    def __init__(self):
        self._orders = []
        self._positions = {}
        self._holdings = {}
        self._cash = 1_000_000

    def login(self):
        return True

    def place_order(
        self,
        ticker,
        side,
        quantity,
        order_type,
        price=None,
    ):
        order = {
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "price": price,
            "status": "FILLED",
        }

        self._orders.append(order)
        return order

    def positions(self):
        return self._positions

    def holdings(self):
        return self._holdings

    def orders(self):
        return self._orders

    def funds(self):
        return {
            "cash": self._cash,
        }
