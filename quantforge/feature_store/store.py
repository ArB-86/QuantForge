import duckdb
import pandas as pd
from pathlib import Path

class FeatureStore:
    def __init__(self, db_path: str = 'data/features.db'):
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))

    def save(self, df: pd.DataFrame):
        print(f'Saving {len(df)} rows to Feature Store...')
        self.conn.register('temp_feat', df)
        self.conn.execute('CREATE OR REPLACE TABLE features AS SELECT * FROM temp_feat')
        self.conn.unregister('temp_feat')
        print('Feature Store updated.')

    def load(self) -> pd.DataFrame:
        return self.conn.execute('SELECT * FROM features ORDER BY Date, Ticker').df()
