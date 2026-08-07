import numpy as np
import pandas as pd

class WalkForwardSplitter:
    def __init__(self, train_days: int = 252, test_days: int = 20, step_days: int = 20):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def split(self, df: pd.DataFrame, date_col: str = 'Date'):
        dates = np.sort(df[date_col].unique())
        total_days = len(dates)
        
        if total_days < self.train_days + self.test_days:
            raise ValueError(f"Not enough days in dataset ({total_days}) for train ({self.train_days}) + test ({self.test_days})")

        splits = []
        for i in range(0, total_days - self.train_days - self.test_days + 1, self.step_days):
            train_dates = set(dates[i : i + self.train_days])
            test_dates = set(dates[i + self.train_days : i + self.train_days + self.test_days])
            
            train_idx = df.index[df[date_col].isin(train_dates)].to_numpy()
            test_idx = df.index[df[date_col].isin(test_dates)].to_numpy()
            
            splits.append((train_idx, test_idx))
            
        return splits
