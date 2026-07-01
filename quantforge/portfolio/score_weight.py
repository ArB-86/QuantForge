import numpy as np
import pandas as pd


def build_score_weight_portfolio(
    df,
    score_column="PRED_RETURN",
    top_n=10,
    max_weight=0.20,
    min_weight=0.02,
):
    """
    Score-weighted portfolio.

    Higher prediction
        =>
    Higher allocation.
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

        scores = (
            picks[score_column]
            .clip(lower=0)
        )

        if scores.sum() == 0:

            weights = np.repeat(
                1 / len(picks),
                len(picks),
            )

        else:

            weights = (
                scores
                / scores.sum()
            )

        weights = np.clip(
            weights,
            min_weight,
            max_weight,
        )

        weights = weights / weights.sum()

        picks["Weight"] = weights

        portfolios.append(
            picks
        )

    return pd.concat(
        portfolios,
        ignore_index=True,
    )
