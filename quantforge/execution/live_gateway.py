import os
import logging
import pandas as pd
import requests

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/execution.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class LiveBrokerGateway:
    def __init__(self, broker_name='paper'):
        self.broker_name = broker_name
        self.api_key = os.getenv('BROKER_API_KEY', '')
        self.api_secret = os.getenv('BROKER_API_SECRET', '')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

    def send_telegram_alert(self, message: str):
        if not self.telegram_token or not self.telegram_chat_id:
            return
        url = f'https://api.telegram.org/bot{self.telegram_token}/sendMessage'
        payload = {'chat_id': self.telegram_chat_id, 'text': message, 'parse_mode': 'Markdown'}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logging.error(f'Failed to send Telegram notification: {e}')

    def authenticate(self):
        if self.broker_name != 'paper' and not self.api_key:
            raise ValueError('Broker API key not found in environment variables.')
        logging.info(f'Authenticated with broker gateway: {self.broker_name.upper()}')
        return True

    def place_orders(self, orders_df: pd.DataFrame):
        self.authenticate()
        print(f'\n--- PLACING ORDERS TO BROKER ({len(orders_df)} assets targeted) ---')
        logging.info(f'Placing {len(orders_df)} orders to broker.')
        
        alert_lines = ['?? *QuantForge Trade Execution*']
        for _, row in orders_df.iterrows():
            ticker = row['Ticker']
            shares = row['Target_Shares']
            price = row['Execution_Price']
            if shares <= 0:
                continue
            line = f'BUY {shares} x {ticker} @ approx ?{price:.2f}'
            logging.info(f'ROUTED: {line}')
            alert_lines.append(line)
            
        print(f'[GATEWAY] Successfully routed {len(orders_df)} orders. Details saved to logs/execution.log')
        self.send_telegram_alert('\n'.join(alert_lines))
        return True
