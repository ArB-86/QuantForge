import numpy as np
import pandas as pd


def winsorize(
    series,
    lower=0.01,
    upper=0.99,
):

    lo = series.quantile(lower)
    hi = series.quantile(upper)

    return series.clip(lo, hi)


def zscore(series):

    std = series.std()

    if std == 0 or np.isnan(std):
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series
        - series.mean()
    ) / std


def normalize_cross_section(
    df,
    values,
):

    out = values.copy()

    for date, idx in (
        df.groupby("Date").groups.items()
    ):

        s = winsorize(
            out.loc[idx]
        )

        out.loc[idx] = zscore(s)

    return out.fillna(0.0)


def rank_cross_section(
    df,
    values,
):

    return (
        values
        .groupby(df["Date"])
        .rank(pct=True)
        .sub(0.5)
        .mul(2.0)
        .fillna(0.0)
    )
