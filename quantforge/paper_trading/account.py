from quantforge.paper_trading.portfolio import Portfolio


class PaperTradingAccount:
    def __init__(self, capital: float = 1_000_000):
        self.portfolio = Portfolio(capital)
