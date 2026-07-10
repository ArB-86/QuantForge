from quantforge.live_trading.paper_broker import PaperBroker
from quantforge.live_trading.market_hours import MarketHours
from quantforge.live_trading.order import LiveOrder


class LiveTradingEngine:
    def __init__(self, broker=None, validate_market_hours=False):
        self.broker = broker or PaperBroker()
        self.validate_market_hours = validate_market_hours

    def login(self):
        return self.broker.login()

    def submit(self, order: LiveOrder):
        if self.validate_market_hours:
            MarketHours.ensure_open()

        return self.broker.place_order(
            ticker=order.ticker,
            side=order.side.value,
            quantity=order.quantity,
            order_type=order.order_type.value,
            price=order.price,
        )

    def positions(self):
        return self.broker.positions()

    def holdings(self):
        return self.broker.holdings()

    def orders(self):
        return self.broker.orders()

    def funds(self):
        return self.broker.funds()
