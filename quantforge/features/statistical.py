import pandas as pd


def add_statistical_features(df):

    g = df.groupby("Ticker")

    df["ROLL_STD20"] = (
        g["LOG_RETURN"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["ROLL_SKEW20"] = (
        g["LOG_RETURN"]
        .rolling(20)
        .skew()
        .reset_index(level=0, drop=True)
    )

    df["ROLL_KURT20"] = (
        g["LOG_RETURN"]
        .rolling(20)
        .kurt()
        .reset_index(level=0, drop=True)
    )

    df["ZSCORE_20D"] = (
        (
            df["RETURN_20D"]
            - g["RETURN_20D"].transform("mean")
        )
        /
        (
            g["RETURN_20D"].transform("std") + 1e-8
        )
    )

    return df
