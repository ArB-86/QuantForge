from quantforge.live_trading.connectors.factory import create_connector


class OrderRouter:
    def __init__(self, connector="paper"):
        self.connector = create_connector(connector)

    def submit(self, order):
        return self.connector.place_order(
            ticker=order.ticker,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
        )

    def orders(self):
        return self.connector.orders()

    def positions(self):
        return self.connector.positions()

    def holdings(self):
        return self.connector.holdings()
