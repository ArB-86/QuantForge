import numpy as np
import pandas as pd

def add_base_features(df):
    df = df.copy()
    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "Adj Close": "Adj_Close",
    }
    df = df.rename(columns=rename)
    g = df.groupby("Ticker", group_keys=False)

    # Returns
    for n in [1,5,20,60,120,250]:
        df[f"RETURN_{n}D"] = g["Close"].pct_change(n, fill_method=None)
    df["LOG_RETURN"] = np.log(df["Close"] / g["Close"].shift(1))

    # Return ranks
    for n in [5,20,60,120,250]:
        df[f"RETURN_{n}D_RANK"] = df.groupby("Date")[f"RETURN_{n}D"].rank(pct=True)

    # Volume rank / size rank
    df["Volume_RANK"] = df.groupby("Date")["Volume"].rank(pct=True)
    df["SIZE_RANK"] = df.groupby("Date")["Close"].rank(pct=True)

    # Volatility
    df["VOL_20D"] = g["LOG_RETURN"].transform(lambda x: x.rolling(20).std())

    # EMAs
    df["EMA20"] = g["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["EMA50"] = g["Close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["EMA200"] = g["Close"].transform(lambda x: x.ewm(span=200, adjust=False).mean())
    df["EMA20_OVER_EMA200"] = df["EMA20"] / df["EMA200"]

    # ATR
    prev_close = g["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = tr.groupby(df["Ticker"]).transform(lambda x: x.rolling(14).mean())
    df["ATR_PCT"] = df["ATR14"] / df["Close"]

    # RSI14
    delta = g["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["RSI14_RANK"] = df.groupby("Date")["RSI14"].rank(pct=True)

    # MACD
    ema12 = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # VWAP
    pv = df["Close"] * df["Volume"]
    df["VWAP"] = pv.groupby(df["Ticker"]).cumsum() / df["Volume"].groupby(df["Ticker"]).cumsum()

    # Bollinger
    ma20 = g["Close"].transform(lambda x: x.rolling(20).mean())
    std20 = g["Close"].transform(lambda x: x.rolling(20).std())
    df["BB_UPPER"] = ma20 + 2 * std20
    df["BB_LOWER"] = ma20 - 2 * std20

    # Market / Liquidity
    df["MARKET_RET_20D"] = df["RETURN_20D"].mean()
    df["DOLLAR_VOLUME"] = df["Close"] * df["Volume"]
    df["LOG_DOLLAR_VOLUME"] = np.log(df["DOLLAR_VOLUME"] + 1)
    df["VOLUME_RATIO_20D"] = df["Volume"] / g["Volume"].rolling(20).mean().reset_index(level=0, drop=True)

    # Additional required columns
    df["PRICE_TO_52W_HIGH"] = (
    df["Close"]
    /
    g["Close"].transform(
        lambda x: x.rolling(252, min_periods=1).max()
    )
)
    df["RETURN_120D_MINUS_20D"] = df["RETURN_120D"] - df["RETURN_20D"]
    df["TARGET_20D_RETURN"] = df["RETURN_20D"].shift(-20)

    return df

