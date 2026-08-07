import os
import pandas as pd
from datetime import datetime, timedelta
from quantforge.live_data.downloader import MarketDataDownloader
from quantforge.dataset.builder import DatasetBuilder

def update_database(tickers_file: str = 'data/tickers.csv', start_date_fallback: str = '2015-01-01'):
    if not os.path.exists(tickers_file):
        raise FileNotFoundError(f'Tickers file not found: {tickers_file}')

    tickers = pd.read_csv(tickers_file)['Ticker'].tolist()
    
    db = DatasetBuilder()
    latest_date = db.get_latest_date()
    
    if latest_date:
        start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = start_date_fallback
        
    today = datetime.now().strftime('%Y-%m-%d')
    
    if pd.Timestamp(start_date) >= pd.Timestamp(today):
        print('Database is already up to date.')
        return

    print(f'Updating {len(tickers)} tickers from {start_date} to {today}...')
    
    dl = MarketDataDownloader(batch_size=50)
    df = dl.download(tickers, start_date=start_date, end_date=today)
    
    db.upsert_data(df)
    print(f'Final DB Date: {db.get_latest_date()}')

if __name__ == '__main__':
    update_database()
