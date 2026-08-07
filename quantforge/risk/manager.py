import pandas as pd

class RiskManager:
    def __init__(self, max_drawdown_limit=0.35):
        self.max_dd_limit = max_drawdown_limit

    def apply_overlay(self, portfolio_weights: pd.DataFrame, market_returns: pd.DataFrame) -> pd.DataFrame:
        # Pass-through for baseline alpha validation
        return portfolio_weights
