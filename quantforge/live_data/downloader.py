import pandas as pd
import yfinance as yf
import os

class DataDownloader:
    def __init__(self, tickers_path='data/tickers.csv', output_path='data/raw_market_data.parquet'):
        self.tickers_path = tickers_path
        self.output_path = output_path

    def download(self, start_date='2020-01-01', end_date=None):
        if not os.path.exists(self.tickers_path):
            raise FileNotFoundError(f"Ticker file not found at {self.tickers_path}")
            
        tickers_df = pd.read_csv(self.tickers_path)
        tickers = tickers_df['Ticker'].tolist()
        print(f"Downloading historical data for {len(tickers)} tickers: {tickers}")
        
        all_data = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    print(f"WARNING: No data returned for {ticker}")
                    continue
                # Handle multi-index columns if returned by yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.reset_index()
                df['Ticker'] = ticker
                all_data.append(df)
            except Exception as e:
                print(f"ERROR downloading {ticker}: {e}")
                
        if not all_data:
            raise RuntimeError("Failed to download data for any tickers.")
            
        master_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        master_df.to_parquet(self.output_path, index=False)
        print(f"Successfully saved raw data for {len(all_data)} tickers to {self.output_path}")

if __name__ == '__main__':
    DataDownloader().download()
