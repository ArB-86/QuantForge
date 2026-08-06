from datetime import datetime

from quantforge.live_trading.router import OrderRouter
from quantforge.live_trading.session import TradingSession
from quantforge.live_trading.pretrade import PreTradeRisk
from quantforge.live_trading.posttrade import PostTradeProcessor
from quantforge.live_trading.audit import AuditLogger
from quantforge.live_trading.metrics import LiveMetrics
from quantforge.live_trading.report import LiveTradingReport
from quantforge.live_trading.observer import Observable


class LiveExecutionEngine(Observable):

    def __init__(self, connector="paper", validate_market_hours=False):
        Observable.__init__(self)
        self.router = OrderRouter(connector)
        self.session = TradingSession()
        self.validate_market_hours = validate_market_hours
        self.risk = PreTradeRisk()
        self.posttrade = PostTradeProcessor()
        self.audit = AuditLogger()
        self.metrics = LiveMetrics()
        self.report = LiveTradingReport(self)

    def execute(self, orders):

        if self.validate_market_hours:
            self.session.ensure_open()

        fills = []

        for order in orders:

            self.risk.validate(order)
            result = self.router.submit(order)

            fill = self.posttrade.record(order, result)
            self.metrics.update(fill)
            self.audit.log(fill)
            self.notify("fill", fill)
            fills.append(fill)

        return fills


    def save_report(self, output_dir="results/live_trading"):
        return self.report.generate(output_dir)
