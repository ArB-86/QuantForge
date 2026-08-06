import pandas as pd
import numpy as np


def add_market_features(df):

    g = df.groupby("Ticker")

    # Relative momentum
    df["RELATIVE_20D"] = (
        df["RETURN_20D"] -
        df["MARKET_RET_20D"]
    )

    # Relative 60D momentum
    df["RELATIVE_60D"] = (
        df["RETURN_60D"] -
        df["MARKET_RET_20D"]
    )

    # Trend persistence
    df["TREND_STRENGTH"] = (
        (
            df["EMA20"] >
            df["EMA50"]
        ).astype(int)
        +
        (
            df["EMA50"] >
            df["EMA200"]
        ).astype(int)
    )

    # ATR relative to recent volatility
    df["ATR_VOL_RATIO"] = (
        df["ATR14"] /
        (df["VOL_20D"] + 1e-8)
    )

    # Rolling volatility regime
    df["VOL_REGIME"] = (
        g["VOL_20D"]
        .transform(
            lambda x:
            x / (
                x.rolling(
                    60,
                    min_periods=20
                ).mean()
                + 1e-8
            )
        )
    )

    # Momentum persistence
    df["MOM_PERSISTENCE"] = (
        (
            df["RETURN_20D"] > 0
        ).astype(int)
        +
        (
            df["RETURN_60D"] > 0
        ).astype(int)
        +
        (
            df["RETURN_120D"] > 0
        ).astype(int)
    )

    return df
