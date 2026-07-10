import math


class PositionSizer:

    def __init__(
        self,
        capital,
        risk_per_trade=0.01,
        max_position_pct=0.10,
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def size(self, signal):

        risk_per_share = abs(signal.entry - signal.stop_loss)

        if risk_per_share <= 0:
            signal.quantity = 0
            signal.position_value = 0
            return signal

        max_risk = self.capital * self.risk_per_trade

        qty = math.floor(max_risk / risk_per_share)

        max_qty = math.floor(
            self.capital *
            self.max_position_pct /
            signal.entry
        )

        qty = min(qty, max_qty)

        signal.quantity = max(qty, 0)
        signal.position_value = signal.quantity * signal.entry

        return signal

    def size_all(self, signals):

        for s in signals:
            if s.action == "BUY":
                self.size(s)

        return signals
