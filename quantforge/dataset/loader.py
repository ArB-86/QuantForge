import os
import pandas as pd
import yfinance as yf

class MarketDataLoader:
    def __init__(self, tickers=None, start_date='2020-01-01', save_path='data/raw_market_data.parquet'):
        self.tickers = tickers or [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'ICICIBANK.NS',
            'INFY.NS', 'SBIN.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS',
            'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'M&M.NS', 'ULTRACEMCO.NS',
            'POWERGRID.NS', 'TITAN.NS', 'BAJAJFINSV.NS', 'KOTAKBANK.NS', 'COALINDIA.NS',
            'ONGC.NS', 'NTPC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'WIPRO.NS',
            'NESTLEIND.NS', 'GRASIM.NS', 'ADANIENT.NS'
        ]
        self.start_date = start_date
        self.save_path = save_path

    def download_all(self) -> pd.DataFrame:
        print(f'Downloading historical data for {len(self.tickers)} tickers...')
        all_data = []
        
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, progress=False)
                if df.empty:
                    print(f'WARNING: No data returned for {ticker}')
                    continue
                
                # Flatten multi-index columns if returned by newer yfinance versions
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                df = df.reset_index()
                df['Ticker'] = ticker
                all_data.append(df)
            except Exception as e:
                print(f'Error downloading {ticker}: {e}')
                
        if not all_data:
            raise RuntimeError('Failed to download data for any tickers.')
            
        combined = pd.concat(all_data, ignore_index=True)
        # Standardize column names
        combined.columns = [str(c).capitalize() if c not in ['Ticker', 'Date'] else c for c in combined.columns]
        if 'Date' not in combined.columns and 'Datetime' in combined.columns:
            combined = combined.rename(columns={'Datetime': 'Date'})
            
        return combined

    def save(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df.to_parquet(self.save_path, index=False)
        print(f'Successfully saved raw data for {df["Ticker"].nunique()} tickers to {self.save_path}')

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f'Raw data file not found at {self.save_path}. Run download_all first.')
        return pd.read_parquet(self.save_path)
