import pandas as pd


def build_equal_weight_portfolio(
    df,
    score_column,
    top_n=10
):
    """
    Build equal-weight long portfolio.

    Parameters
    ----------
    df : DataFrame

    score_column : str

    top_n : int

    Returns
    -------
    DataFrame
    """

    portfolio = []

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

        picks["Weight"] = 1.0 / top_n

        portfolio.append(picks)

    return pd.concat(
        portfolio,
        ignore_index=True
    )
