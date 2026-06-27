import pandas as pd


def run_backtest(
    portfolio_df,
    return_column="TARGET_5D_RETURN",
    holding_days=5
):
    """
    Generic walk-forward backtest.

    Parameters
    ----------
    portfolio_df : DataFrame

    return_column : str

    holding_days : int

    Returns
    -------
    DataFrame
    """

    portfolio_df = portfolio_df.copy()

    portfolio_df["Date"] = pd.to_datetime(
        portfolio_df["Date"]
    )

    rebalance_dates = sorted(
        portfolio_df["Date"].unique()
    )

    rebalance_dates = rebalance_dates[
        ::holding_days
    ]

    portfolio_returns = []

    for date in rebalance_dates:

        g = portfolio_df[
            portfolio_df["Date"] == date
        ]

        if len(g) == 0:
            continue

        portfolio_returns.append({

            "Date": date,

            "Return":
            g[
                return_column
            ].mean()

        })

    portfolio = pd.DataFrame(
        portfolio_returns
    )

    portfolio["Equity"] = (

        1 +

        portfolio["Return"]

    ).cumprod()

    return portfolio