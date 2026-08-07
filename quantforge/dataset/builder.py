import os
import duckdb
import pandas as pd
from pathlib import Path

class DatasetBuilder:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "market_data.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_db()

    def _init_db(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                Date TIMESTAMP,
                Ticker VARCHAR,
                Open FLOAT,
                High FLOAT,
                Low FLOAT,
                Close FLOAT,
                Adj_Close FLOAT,
                Volume BIGINT,
                PRIMARY KEY (Date, Ticker)
            )
        ''')

    def upsert_data(self, df: pd.DataFrame):
        print(f"Upserting {len(df)} rows to DuckDB...")
        # Create temporary table from dataframe
        self.conn.register('temp_df', df)
        
        # Insert or ignore duplicates
        self.conn.execute('''
            INSERT INTO ohlcv 
            SELECT * FROM temp_df
            ON CONFLICT (Date, Ticker) DO UPDATE SET
                Open = EXCLUDED.Open,
                High = EXCLUDED.High,
                Low = EXCLUDED.Low,
                Close = EXCLUDED.Close,
                Adj_Close = EXCLUDED.Adj_Close,
                Volume = EXCLUDED.Volume
        ''')
        self.conn.unregister('temp_df')
        print("Upsert complete.")

    def get_latest_date(self) -> pd.Timestamp:
        result = self.conn.execute('SELECT MAX(Date) FROM ohlcv').fetchone()[0]
        return pd.Timestamp(result) if result else None
