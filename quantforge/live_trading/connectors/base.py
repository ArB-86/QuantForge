from abc import ABC, abstractmethod


class BrokerConnector(ABC):

    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def place_order(self, *args, **kwargs):
        ...

    @abstractmethod
    def modify_order(self, *args, **kwargs):
        ...

    @abstractmethod
    def cancel_order(self, *args, **kwargs):
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
