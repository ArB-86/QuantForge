from quantforge.signals.engine import SignalEngine
from quantforge.signals.allocator import AllocationEngine
from quantforge.signals.position_sizer import PositionSizer
from quantforge.signals.trade_sheet import TradeSheetGenerator


class DailyTradingPipeline:

    def __init__(
        self,
        capital=1_000_000,
    ):
        self.signal_engine = SignalEngine()
        self.allocator = AllocationEngine()
        self.position_sizer = PositionSizer(capital)
        self.trade_sheet = TradeSheetGenerator()

    def run(self, predictions):

        signals = self.signal_engine.generate(predictions)

        signals = self.allocator.allocate(signals)

        signals = self.position_sizer.size_all(signals)

        return self.trade_sheet.generate(signals)
