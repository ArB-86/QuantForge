class RiskManager:
    def __init__(self,
                 max_position_pct=0.10,
                 max_notional=1000000):
        self.max_position_pct = max_position_pct
        self.max_notional = max_notional

    def validate(self, cash, portfolio_value, quantity, price):
        notional = quantity * price

        if notional > self.max_notional:
            raise ValueError("Max notional exceeded")

        if portfolio_value > 0 and notional / portfolio_value > self.max_position_pct:
            raise ValueError("Max position exceeded")

        if notional > cash:
            raise ValueError("Insufficient cash")

        return True
