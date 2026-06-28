import numpy as np


def add_volatility_features(df):

    # Parkinson Volatility
    df["PARKINSON_VOL"] = np.sqrt(
        (
            np.log(df["High"] / df["Low"]) ** 2
        ) / (4 * np.log(2))
    )

    # Garman-Klass Volatility
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    df["GARMAN_KLASS_VOL"] = np.sqrt(
        0.5 * log_hl**2
        - (2 * np.log(2) - 1) * log_co**2
    )

    # Daily Range
    df["TRUE_RANGE"] = (
        df["High"] - df["Low"]
    ) / df["Close"]

    # ATR Normalized
    df["ATR_NORMALIZED"] = (
        df["ATR14"] /
        df["Close"]
    )

    # Bollinger Width
    df["BB_WIDTH"] = (
        df["BB_UPPER"] -
        df["BB_LOWER"]
    ) / df["Close"]

    return df
