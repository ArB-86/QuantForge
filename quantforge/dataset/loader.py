import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path='data/raw_market_data.parquet'):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Raw data file not found at {self.file_path}. Run downloader first.")
        df = pd.read_parquet(self.file_path)
        # Ensure standard column names
        rename_map = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj_Close', 'Volume': 'Volume', 'Date': 'Date'}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df
