import pandas as pd
import numpy as np


def add_alpha_factors(df):

    df = df.sort_values(
        ["Ticker", "Date"]
    )

    g = df.groupby("Ticker")

    # --------------------
    # Residual Momentum
    # --------------------

    df["RESIDUAL_MOMENTUM"] = (
        df["RETURN_20D"]
        -
        df["MARKET_RET_20D"]
    )

    # --------------------
    # Momentum Acceleration
    # --------------------

    df["MOMENTUM_ACCEL"] = (
        df["RETURN_20D"]
        -
        df["RETURN_60D"]
    )

    # --------------------
    # Volatility Compression
    # --------------------

    df["VOL_COMPRESSION"] = (

        df["VOL_20D"]

        /

        g["VOL_20D"]
        .rolling(60)
        .mean()
        .reset_index(level=0,drop=True)

    )

    # --------------------
    # Dollar Volume
    # --------------------

    df["DOLLAR_VOLUME"] = (
        df["Close"]
        *
        df["Volume"]
    )

    # --------------------
    # Liquidity Rank
    # --------------------

    df["LIQUIDITY_RANK"] = (

        df.groupby("Date")["DOLLAR_VOLUME"]

        .rank(pct=True)

    )

    # --------------------
    # Volatility Rank
    # --------------------

    df["VOLATILITY_RANK"] = (

        df.groupby("Date")["VOL_20D"]

        .rank(pct=True)

    )

    return dfimport pandas as pd
import numpy as np


def add_alpha_factors(df):

    df = df.sort_values(
        ["Ticker", "Date"]
    )

    g = df.groupby("Ticker")

    # --------------------
    # Residual Momentum
    # --------------------

    df["RESIDUAL_MOMENTUM"] = (
        df["RETURN_20D"]
        -
        df["MARKET_RET_20D"]
    )

    # --------------------
    # Momentum Acceleration
    # --------------------

    df["MOMENTUM_ACCEL"] = (
        df["RETURN_20D"]
        -
        df["RETURN_60D"]
    )

    # --------------------
    # Volatility Compression
    # --------------------

    df["VOL_COMPRESSION"] = (

        df["VOL_20D"]

        /

        g["VOL_20D"]
        .rolling(60)
        .mean()
        .reset_index(level=0,drop=True)

    )

    # --------------------
    # Dollar Volume
    # --------------------

    df["DOLLAR_VOLUME"] = (
        df["Close"]
        *
        df["Volume"]
    )

    # --------------------
    # Liquidity Rank
    # --------------------

    df["LIQUIDITY_RANK"] = (

        df.groupby("Date")["DOLLAR_VOLUME"]

        .rank(pct=True)

    )

    # --------------------
    # Volatility Rank
    # --------------------

    df["VOLATILITY_RANK"] = (

        df.groupby("Date")["VOL_20D"]

        .rank(pct=True)

    )

    return df