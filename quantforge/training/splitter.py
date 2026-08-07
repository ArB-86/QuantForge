import numpy as np
import pandas as pd

class WalkForwardSplitter:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, df: pd.DataFrame):
        if 'Date' not in df.columns:
            raise KeyError("DataFrame must contain a 'Date' column for walk-forward splitting.")
            
        dates = df['Date'].drop_duplicates().sort_values().reset_index(drop=True)
        total_dates = len(dates)
        fold_size = total_dates // (self.n_splits + 1)
        
        splits = []
        for i in range(1, self.n_splits + 1):
            train_end_idx = i * fold_size
            test_end_idx = min((i + 1) * fold_size, total_dates)
            
            train_dates = dates.iloc[:train_end_idx]
            test_dates = dates.iloc[train_end_idx:test_end_idx]
            
            if test_dates.empty:
                continue
                
            train_idx = df[df['Date'].isin(train_dates)].index.values
            test_idx = df[df['Date'].isin(test_dates)].index.values
            
            splits.append((train_idx, test_idx))
            
        return splits
