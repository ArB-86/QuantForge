from quantforge.paper_trading.account import PaperTradingAccount
from quantforge.paper_trading.broker import PaperBroker
from quantforge.paper_trading.execution import ExecutionEngine
from quantforge.paper_trading.market import Market
from quantforge.paper_trading.order import Order
from quantforge.paper_trading.types import OrderSide
from quantforge.paper_trading.observer import Observable
from quantforge.paper_trading.performance import PerformanceTracker
from quantforge.paper_trading.ledger import Ledger
from quantforge.paper_trading.report import ReportGenerator
from quantforge.paper_trading.replay import ReplayEngine
from quantforge.paper_trading.risk import RiskManager


class PaperTradingEngine(Observable):
    def __init__(self, capital=1_000_000):
        super().__init__()

        self.account = PaperTradingAccount(capital)
        self.broker = PaperBroker(self.account)
        self.execution = ExecutionEngine(self.broker)

        self.market = Market()
        self.risk = RiskManager()
        self.performance = PerformanceTracker()
        self.ledger = Ledger()
        self.report = ReportGenerator(self)
        self.replay = ReplayEngine(self)

    def submit(self, ticker, side, quantity, price):
        order = Order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=price,
        )

        filled = self.execution.execute([order])[0]

        self.ledger.record(
            "LIVE",
            ticker,
            side.name,
            quantity,
            price,
        )

        self.notify_trade(self, filled)

        return filled

    def buy(self, ticker, quantity, price):
        return self.submit(ticker, OrderSide.BUY, quantity, price)

    def sell(self, ticker, quantity, price):
        return self.submit(ticker, OrderSide.SELL, quantity, price)

    def portfolio_value(self, prices):
        return self.account.portfolio.equity(prices)

    def positions(self):
        return self.account.portfolio.positions

    def cash(self):
        return self.account.portfolio.cash

    def mark(self, prices):
        self.market.update_many(prices)
        return self.portfolio_value(self.market.snapshot())

    def snapshot(self, timestamp):
        equity = self.portfolio_value(self.market.snapshot())

        self.performance.record(
            timestamp,
            equity,
            self.cash(),
        )

        snap = self.performance.latest()
        self.notify_snapshot(self, snap)
        return snap

    def save_report(self, output_dir="results/paper_trading"):
        self.report.generate(output_dir)
