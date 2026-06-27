import pandas as pd


def _cap_weights(
    weights,
    max_weight,
):
    """
    Iteratively cap weights and redistribute excess.
    """

    weights = weights.copy()

    while True:

        excess = (
            weights - max_weight
        ).clip(lower=0).sum()

        if excess <= 1e-12:
            break

        capped = weights > max_weight
        weights[capped] = max_weight

        eligible = weights < max_weight

        if eligible.sum() == 0:
            break

        redistribute = (
            weights[eligible]
            / weights[eligible].sum()
        )

        weights.loc[eligible] += (
            redistribute * excess
        )

    return weights / weights.sum()


def build_inverse_volatility_portfolio(
    df,
    score_column="ENSEMBLE_SCORE",
    volatility_column="VOL_20D",
    top_n=10,
    min_volatility=1e-3,
    max_weight=0.20,
):

    portfolios = []

    for _, group in df.groupby("Date"):

        picks = (
            group
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(top_n)
            .copy()
        )

        inv_vol = (
            1.0
            /
            picks[volatility_column]
            .clip(lower=min_volatility)
        )

        weights = (
            inv_vol
            /
            inv_vol.sum()
        )

        picks["Weight"] = _cap_weights(
            weights,
            max_weight,
        )

        portfolios.append(
            picks
        )

    return pd.concat(
        portfolios,
        ignore_index=True,
    )
