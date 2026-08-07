import pandas as pd
import numpy as np

class BacktestMetrics:
    @staticmethod
    def calculate(results_df: pd.DataFrame):
        returns = results_df['Net_Return']
        equity = results_df['Equity_Curve']
        
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annualized_return) / volatility if volatility != 0 else 0
        
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'Total Return (%)': round(total_return * 100, 2),
            'Annualized Return (%)': round(annualized_return * 100, 2),
            'Annualized Volatility (%)': round(volatility * 100, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
            'Avg Daily Turnover (%)': round(results_df['Turnover'].mean() * 100, 2)
        }
