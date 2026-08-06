from quantforge.paper_trading.broker import PaperBroker


class ExecutionEngine:
    def __init__(self, broker: PaperBroker):
        self.broker = broker

    def execute(self, orders):
        return [self.broker.submit(o) for o in orders]
