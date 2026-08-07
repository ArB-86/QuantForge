import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Generating features with 5-Day Target...')
        df = df.sort_values(['Ticker', 'Date']).copy()
        
        def apply_indicators(g):
            g['RET_1D'] = g['Close'].pct_change(1)
            g['RET_5D'] = g['Close'].pct_change(5)
            g['VOL_20D'] = g['RET_1D'].rolling(20).std() * np.sqrt(252)
            g['LOG_DOLLAR_VOLUME'] = np.log((g['Close'] * g['Volume']) + 1)
            
            ema12 = g['Close'].ewm(span=12, adjust=False).mean()
            ema26 = g['Close'].ewm(span=26, adjust=False).mean()
            g['MACD'] = ema12 - ema26
            g['MACD_SIGNAL'] = g['MACD'].ewm(span=9, adjust=False).mean()
            
            sma20 = g['Close'].rolling(20).mean()
            std20 = g['Close'].rolling(20).std()
            g['BB_UPPER'] = sma20 + (2 * std20)
            g['BB_LOWER'] = sma20 - (2 * std20)
            g['BB_WIDTH'] = (g['BB_UPPER'] - g['BB_LOWER']) / sma20
            
            high_low = g['High'] - g['Low']
            high_close = np.abs(g['High'] - g['Close'].shift())
            low_close = np.abs(g['Low'] - g['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            g['ATR_14'] = tr.rolling(14).mean()
            g['ATR_PCT'] = g['ATR_14'] / g['Close']
            
            delta = g['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
            rs = gain / loss
            g['RSI_14'] = 100 - (100 / (1 + rs))
            
            g['TARGET_1D'] = g['RET_1D'].shift(-1)
            g['TARGET_5D'] = g['Close'].shift(-5) / g['Close'] - 1
            return g

        df = df.groupby('Ticker', group_keys=False).apply(apply_indicators, include_groups=False)
        
        base_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'TARGET_1D', 'TARGET_5D']
        self.feature_columns = [c for c in df.columns if c not in base_cols]
        
        df = df.dropna(subset=self.feature_columns + ['TARGET_5D']).reset_index(drop=True)
        for col in self.feature_columns + ['TARGET_1D', 'TARGET_5D']:
            df[col] = df[col].astype(np.float32)
            
        return df

