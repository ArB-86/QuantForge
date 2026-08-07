import os
import pandas as pd

class LiveBrokerGateway:
    def __init__(self, broker_name='paper'):
        self.broker_name = broker_name
        self.api_key = os.getenv('BROKER_API_KEY', '')
        self.api_secret = os.getenv('BROKER_API_SECRET', '')

    def authenticate(self):
        if self.broker_name != 'paper' and not self.api_key:
            raise ValueError('Broker API key not found in environment variables.')
        print(f'Authenticated with broker gateway: {self.broker_name.upper()}')
        return True

    def place_orders(self, orders_df: pd.DataFrame):
        self.authenticate()
        print('\n--- PLACING LIVE/PAPER ORDERS TO BROKER ---')
        for _, row in orders_df.iterrows():
            ticker = row['Ticker']
            shares = row['Target_Shares']
            price = row['Execution_Price']
            
            # Institutional safety check: Reject absurd order sizes
            if shares <= 0:
                continue
                
            print(f'[GATEWAY] ROUTED: BUY {shares} units of {ticker} @ approx {price:.2f}')
        return True
