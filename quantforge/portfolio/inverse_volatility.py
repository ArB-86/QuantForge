import pandas as pd


def build_inverse_volatility_portfolio(
    df,
    score_column="ENSEMBLE_SCORE",
    volatility_column="VOL_20D",
    top_n=10
):

    portfolios = []

    for date, g in df.groupby("Date"):

        picks = (
            g
            .sort_values(
                score_column,
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        inv_vol = 1.0 / (
            picks[volatility_column]
            .clip(lower=1e-6)
        )

        picks["Weight"] = (
            inv_vol
            /
            inv_vol.sum()
        )

        portfolios.append(
            picks
        )

    return pd.concat(
        portfolios,
        ignore_index=True
    )
