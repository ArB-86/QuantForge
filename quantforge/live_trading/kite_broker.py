from quantforge.live_trading.broker import Broker


class KiteBroker(Broker):
    def __init__(
        self,
        api_key,
        access_token,
    ):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = None

    def login(self):
        from kiteconnect import KiteConnect

        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        return True

    def place_order(
        self,
        ticker,
        side,
        quantity,
        order_type,
        price=None,
    ):
        kwargs = dict(
            variety="regular",
            exchange="NSE",
            tradingsymbol=ticker.replace(".NS", ""),
            transaction_type=side,
            quantity=int(quantity),
            product="CNC",
            order_type=order_type,
        )

        if price is not None:
            kwargs["price"] = price

        return self.kite.place_order(**kwargs)

    def positions(self):
        return self.kite.positions()

    def holdings(self):
        return self.kite.holdings()

    def orders(self):
        return self.kite.orders()

    def funds(self):
        return self.kite.margins()
