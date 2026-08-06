from quantforge.live_trading.connectors.base import BrokerConnector
from quantforge.live_trading.paper_broker import PaperBroker


class PaperConnector(BrokerConnector):

    def __init__(self):
        self.broker = PaperBroker()

    def connect(self):
        return self.broker.login()

    def disconnect(self):
        return True

    def place_order(self, *args, **kwargs):
        return self.broker.place_order(*args, **kwargs)

    def modify_order(self, *args, **kwargs):
        raise NotImplementedError

    def cancel_order(self, *args, **kwargs):
        raise NotImplementedError

    def positions(self):
        return self.broker.positions()

    def holdings(self):
        return self.broker.holdings()

    def orders(self):
        return self.broker.orders()
