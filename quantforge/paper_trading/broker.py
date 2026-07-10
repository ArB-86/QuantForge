from quantforge.paper_trading.order import Order
from quantforge.paper_trading.types import OrderStatus


class PaperBroker:
    def __init__(self, account):
        self.account = account
        self.orders = []

    def submit(self, order: Order):
        if order.side.name == "BUY":
            self.account.portfolio.buy(
                order.ticker,
                order.quantity,
                order.price,
            )
        else:
            self.account.portfolio.sell(
                order.ticker,
                order.quantity,
                order.price,
            )

        order.status = OrderStatus.FILLED
        self.orders.append(order)
        return order
