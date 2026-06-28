import numpy as np


def add_momentum_features(df):

    df["MOM_20_5"] = (
        df["RETURN_20D"] -
        df["RETURN_5D"]
    )

    df["MOM_60_20"] = (
        df["RETURN_60D"] -
        df["RETURN_20D"]
    )

    df["MOM_120_60"] = (
        df["RETURN_120D"] -
        df["RETURN_60D"]
    )

    df["MOM_250_120"] = (
        df["RETURN_250D"] -
        df["RETURN_120D"]
    )

    df["ACCELERATION"] = (
        df["RETURN_20D"] -
        df["RETURN_5D"]
    )

    df["LONG_ACCELERATION"] = (
        df["RETURN_120D"] -
        df["RETURN_20D"]
    )

    return df