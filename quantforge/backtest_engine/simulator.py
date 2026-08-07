import pandas as pd
import numpy as np

class BacktestSimulator:
    def __init__(self, transaction_cost_bps=10.0, stt_bps=1.0):
        # 10 bps execution cost + 1 bp STT on sell side
        self.transaction_cost_rate = transaction_cost_bps / 10000.0
        self.stt_rate = stt_bps / 10000.0

    def run(self, portfolio_weights: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        weights = portfolio_weights.pivot(index='Date', columns='Ticker', values='Weight').fillna(0)
        rets = market_data.pivot(index='Date', columns='Ticker', values='RET_1D').fillna(0)
        
        common_dates = weights.index.intersection(rets.index)
        weights = weights.loc[common_dates]
        rets = rets.loc[common_dates]
        
        # Calculate portfolio turnover
        turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
        
        # Gross returns
        gross_rets = (weights.shift(1).fillna(0) * rets).sum(axis=1)
        
        # Realistic transaction costs applied on turnover
        costs = turnover * self.transaction_cost_rate
        net_rets = gross_rets - costs
        
        # Equity curve starting at 100,000
        equity_curve = 100000 * (1 + net_rets).cumprod()
        
        results = pd.DataFrame({
            'Gross_Return': gross_rets,
            'Turnover': turnover,
            'Net_Return': net_rets,
            'Equity_Curve': equity_curve
        }, index=common_dates)
        
        return results
