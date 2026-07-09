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
    Score-weighted portfolio with volatility adjustment and filters.

    Higher prediction => Higher allocation.
    """
    portfolios = []

    for date, g in df.groupby("Date"):

        # Liquidity filter
        g = g[
            g["LOG_DOLLAR_VOLUME"] >=
            g["LOG_DOLLAR_VOLUME"].quantile(0.30)
        ]

        # ATR filter
        g = g[
            g["ATR_PCT"] <= 0.08
        ]

        # Safety check
        if len(g) == 0:
            continue

        picks = (
            g
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(top_n)
            .copy()
        )

        # Volatility-adjusted scoring
        scores = picks[score_column].clip(lower=0)

        vol = (
            picks["VOL_20D"]
            .fillna(picks["VOL_20D"].median())
            .clip(lower=1e-6)
        )

        scores = scores / vol

        if scores.sum() == 0:
            weights = np.repeat(
                1 / len(picks),
                len(picks),
            )
        else:
            weights = scores / scores.sum()

        # Clipping code unchanged
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