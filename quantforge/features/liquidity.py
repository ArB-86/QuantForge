import numpy as np


def add_liquidity_features(df):

    df["ILLIQUIDITY"] = (
        df["LOG_RETURN"].abs()
        /
        (df["DOLLAR_VOLUME"] + 1)
    )

    df["DOLLAR_VOLUME_LOG_DIFF"] = (
        df["LOG_DOLLAR_VOLUME"] -
        df["LOG_DOLLAR_VOLUME"].rolling(20).mean()
    )

    df["VOLUME_SHOCK"] = (
        df["VOLUME_RATIO_20D"] - 1.0
    )

    return df