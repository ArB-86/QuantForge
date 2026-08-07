import time
import pandas as pd
import yfinance as yf
from quantforge.dataset.schema import enforce_schema

class MarketDataDownloader:
    def __init__(self, batch_size: int = 50, retries: int = 3):
        self.batch_size = batch_size
        self.retries = retries

    def download(self, tickers: list[str], start_date: str, end_date: str = None) -> pd.DataFrame:
        print(f"Starting download for {len(tickers)} tickers from {start_date}")
        all_frames = []
        
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            print(f"Processing batch {(i // self.batch_size) + 1} ({len(batch)} symbols)")
            
            df = self._download_batch(batch, start_date, end_date)
            if df is not None and not df.empty:
                all_frames.append(df)
                
            time.sleep(1)

        if not all_frames:
            raise RuntimeError("No data downloaded.")

        master_df = pd.concat(all_frames, ignore_index=True)
        master_df = enforce_schema(master_df)
        return master_df

    def _download_batch(self, batch: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        for attempt in range(self.retries):
            try:
                df = yf.download(
                    tickers=batch, start=start_date, end=end_date,
                    auto_adjust=False, group_by="ticker", progress=False
                )
                if not df.empty:
                    return self._reshape(df, batch)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return None

    def _reshape(self, df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0, future_stack=True).reset_index()
            df = df.rename(columns={"level_1": "Ticker", "ticker": "Ticker"})
        else:
            df = df.reset_index()
            df['Ticker'] = batch[0]
            
        df = df.rename(columns={"Adj Close": "Adj_Close", "Datetime": "Date"})
        df = df.dropna(subset=["Close"]).copy()
        return df
