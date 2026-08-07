import pandas as pd
import numpy as np

OHLCV_SCHEMA = {
    'Date': 'datetime64[ns]',
    'Ticker': 'string',
    'Open': np.float32,
    'High': np.float32,
    'Low': np.float32,
    'Close': np.float32,
    'Adj_Close': np.float32,
    'Volume': np.int64
}

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(OHLCV_SCHEMA.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    df = df[list(OHLCV_SCHEMA.keys())].copy()

    for col, dtype in OHLCV_SCHEMA.items():
        if col == 'Date':
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)

    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    return df
