import numpy as np


def add_trend_features(df):

    df["EMA20_DISTANCE"] = (
        (df["Close"] - df["EMA20"]) /
        df["EMA20"]
    )

    df["EMA50_DISTANCE"] = (
        (df["Close"] - df["EMA50"]) /
        df["EMA50"]
    )

    df["EMA200_DISTANCE"] = (
        (df["Close"] - df["EMA200"]) /
        df["EMA200"]
    )

    df["EMA20_50_SPREAD"] = (
        (df["EMA20"] - df["EMA50"]) /
        df["EMA50"]
    )

    df["EMA50_200_SPREAD"] = (
        (df["EMA50"] - df["EMA200"]) /
        df["EMA200"]
    )

    df["PRICE_VS_VWAP"] = (
        (df["Close"] - df["VWAP"]) /
        df["VWAP"]
    )

    return df
