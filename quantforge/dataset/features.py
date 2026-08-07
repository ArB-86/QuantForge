import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
            'RET_1D', 'RET_5D', 'VOL_20D', 'LOG_DOLLAR_VOLUME',
            'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER', 'BB_WIDTH',
            'ATR_14', 'ATR_PCT', 'RSI_14'
        ]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        print('Generating features with 5-Day Target...')
        if isinstance(df.index, pd.MultiIndex) or 'Ticker' in df.index.names or 'Date' in df.index.names:
            df = df.reset_index()
            
        for col in ['index', 'level_0', 'level_1']:
            if col in df.columns:
                df = df.drop(columns=[col])

        if 'Ticker' not in df.columns or 'Date' not in df.columns:
            raise KeyError("DataFrame must contain 'Ticker' and 'Date' columns.")

        processed_dfs = []
        for ticker, group in df.groupby('Ticker'):
            g = group.sort_values('Date').copy()
            
            # Returns
            g['RET_1D'] = g['Close'].pct_change(1)
            g['RET_5D'] = g['Close'].pct_change(5)
            
            # Volatility
            g['VOL_20D'] = g['RET_1D'].rolling(20).std() * np.sqrt(252)
            
            # Dollar Volume
            g['LOG_DOLLAR_VOLUME'] = np.log(g['Close'] * g['Volume'] + 1)
            
            # MACD
            ema12 = g['Close'].ewm(span=12, adjust=False).mean()
            ema26 = g['Close'].ewm(span=26, adjust=False).mean()
            g['MACD'] = ema12 - ema26
            g['MACD_SIGNAL'] = g['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            ma20 = g['Close'].rolling(20).mean()
            std20 = g['Close'].rolling(20).std()
            g['BB_UPPER'] = ma20 + (2 * std20)
            g['BB_LOWER'] = ma20 - (2 * std20)
            g['BB_WIDTH'] = (g['BB_UPPER'] - g['BB_LOWER']) / ma20
            
            # ATR
            high_low = g['High'] - g['Low']
            high_close = (g['High'] - g['Close'].shift()).abs()
            low_close = (g['Low'] - g['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            g['ATR_14'] = tr.rolling(14).mean()
            g['ATR_PCT'] = g['ATR_14'] / g['Close']
            
            # RSI
            delta = g['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            g['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Targets
            g['TARGET_1D'] = g['Close'].shift(-1) / g['Close'] - 1.0
            g['TARGET_5D'] = g['Close'].shift(-5) / g['Close'] - 1.0
            
            processed_dfs.append(g)
            
        res = pd.concat(processed_dfs, ignore_index=True)
        res = res.dropna().reset_index(drop=True)
        return res
