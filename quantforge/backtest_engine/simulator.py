import pandas as pd
import numpy as np

class BacktestSimulator:
    def __init__(self, initial_capital=100000, transaction_cost=0.0015):
        self.initial_capital = initial_capital
        self.tc = transaction_cost

    def run(self, portfolio_df: pd.DataFrame, market_data_df: pd.DataFrame) -> pd.DataFrame:
        weights = portfolio_df.pivot(index='Date', columns='Ticker', values='Weight').fillna(0)
        returns = market_data_df.pivot(index='Date', columns='Ticker', values='RET_1D').fillna(0)
        
        # Align dates
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Shift weights to prevent lookahead (weight calculated on T, earns return on T+1)
        shifted_weights = weights.shift(1).fillna(0)
        
        # Calculate gross daily return
        gross_pnl = (shifted_weights * returns).sum(axis=1)
        
        # Calculate turnover and transaction costs
        weight_changes = shifted_weights.diff().fillna(shifted_weights)
        turnover = weight_changes.abs().sum(axis=1)
        tc_pnl = turnover * self.tc
        
        # Net daily return
        net_pnl = gross_pnl - tc_pnl
        
        results = pd.DataFrame({
            'Gross_Return': gross_pnl,
            'Turnover': turnover,
            'Net_Return': net_pnl,
            'Equity_Curve': (1 + net_pnl).cumprod() * self.initial_capital
        })
        return results
