import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = ['RET_1D', 'VOL_20D', 'MA_RATIO', 'MOM_5D']

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Generating features and forward targets...')
        df = df.copy()
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # Calculate technical features per ticker
        df['RET_1D'] = df.groupby('Ticker')['Close'].pct_change()
        df['VOL_20D'] = df.groupby('Ticker')['RET_1D'].transform(lambda x: x.rolling(20).std())
        df['MA_RATIO'] = df.groupby('Ticker')['Close'].transform(lambda x: x / x.rolling(50).mean())
        df['MOM_5D'] = df.groupby('Ticker')['Close'].pct_change(5)
        
        # Generate 5-day forward target (will be NaN for the last 5 days)
        df['TARGET_5D'] = df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-5) / x - 1.0)
        
        # Drop rows where core features are NaN, but KEEP rows where TARGET_5D is NaN (needed for live inference)
        df = df.dropna(subset=self.feature_columns + ['Close', 'Volume'])
        
        return df
