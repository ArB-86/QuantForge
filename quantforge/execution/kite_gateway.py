import os
import pandas as pd
from kiteconnect import KiteConnect

class KiteBrokerGateway:
    def __init__(self):
        self.api_key = os.getenv('KITE_API_KEY', '')
        self.access_token = os.getenv('KITE_ACCESS_TOKEN', '')
        if not self.api_key or not self.access_token:
            raise ValueError('Kite API key or access token missing from environment variables.')
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)

    def authenticate(self):
        try:
            profile = self.kite.profile()
            print(f'Successfully authenticated with Zerodha Kite. User: {profile.get("user_id")}')
            return True
        except Exception as e:
            print(f'Kite Authentication Failed: {e}')
            return False

    def place_orders(self, orders_df: pd.DataFrame):
        if not self.authenticate():
            raise ConnectionError('Cannot place orders: Broker authentication failed.')
            
        print('\n--- PLACING LIVE ORDERS VIA KITE CONNECT ---')
        for _, row in orders_df.iterrows():
            ticker = row['Ticker']
            shares = int(row['Target_Shares'])
            if shares <= 0:
                continue
                
            try:
                # Example order placement structure for NSE equity
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=ticker.replace('.NS', ''),
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                    quantity=shares,
                    product=self.kite.PRODUCT_CNC,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                print(f'[KITE GATEWAY] Placed BUY order for {shares} of {ticker}. Order ID: {order_id}')
            except Exception as e:
                print(f'[KITE ERROR] Failed to place order for {ticker}: {e}')
        return True
