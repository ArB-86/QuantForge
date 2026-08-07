import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Optional

class DataLoader:
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run update_dataset first.")

    def load(self, start_date: Optional[str] = None, end_date: Optional[str] = None, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        query = "SELECT * FROM ohlcv WHERE 1=1"
        
        if start_date:
            query += f" AND Date >= '{start_date}'"
        if end_date:
            query += f" AND Date <= '{end_date}'"
        if tickers:
            tickers_str = ", ".join([f"'{t}'" for t in tickers])
            query += f" AND Ticker IN ({tickers_str})"
            
        query += " ORDER BY Date, Ticker"
        
        with duckdb.connect(str(self.db_path), read_only=True) as conn:
            df = conn.execute(query).df()
            
        return df
