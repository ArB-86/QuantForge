from abc import ABC, abstractmethod


class Broker(ABC):

    @abstractmethod
    def login(self):
        ...

    @abstractmethod
    def place_order(
        self,
        ticker,
        side,
        quantity,
        order_type,
        price=None,
    ):
        ...

    @abstractmethod
    def positions(self):
        ...

    @abstractmethod
    def holdings(self):
        ...

    @abstractmethod
    def orders(self):
        ...

    @abstractmethod
    def funds(self):
        ...
