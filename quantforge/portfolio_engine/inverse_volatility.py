import numpy as np
import pandas as pd


def build_inverse_volatility_portfolio(
    df,
    score_column="PRED_RETURN",
    volatility_column="VOL_20D",
    top_n=10,
    max_weight=0.20,
    min_weight=0.02,
):
    """
    Build inverse-volatility weighted portfolio
    independently for each rebalance date.
    """

    portfolios = []

    for date, g in df.groupby("Date"):

        picks = (
            g
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(top_n)
            .copy()
        )

        inv_vol = 1.0 / (
            picks[volatility_column]
            .clip(lower=1e-6)
        )

        weights = inv_vol / inv_vol.sum()

        weights = np.clip(
            weights,
            min_weight,
            max_weight,
        )

        weights = weights / weights.sum()

        picks["Weight"] = weights

        portfolios.append(picks)

    return pd.concat(
        portfolios,
        ignore_index=True,
    )