import os
import pandas as pd

class FeatureStore:
    def __init__(self, store_path='data/feature_store.parquet'):
        self.store_path = store_path

    def save(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        df.to_parquet(self.store_path, index=False)
        print(f'Saving {len(df)} rows to Feature Store...')
        print('Feature Store updated.')

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.store_path):
            raise FileNotFoundError(f'Feature store not found at {self.store_path}.')
        return pd.read_parquet(self.store_path)
