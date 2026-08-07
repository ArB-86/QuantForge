import pandas as pd

class PaperBroker:
    def __init__(self, initial_cash=1000000.0):
        self.cash = initial_cash
        self.positions = {}

    def execute_orders(self, orders_df: pd.DataFrame):
        print('\n--- EXECUTING PAPERTRAINING ORDERS ---')
        for _, row in orders_df.iterrows():
            ticker = row['Ticker']
            shares = row['Target_Shares']
            price = row['Execution_Price']
            cost = shares * price
            
            self.positions[ticker] = shares
            print(f'Executed: BUY {shares} shares of {ticker} at {price:.2f} | Cost: {cost:.2f}')
        print(f'Paper Broker Cash Remaining: {self.cash:,.2f}')
        return True
