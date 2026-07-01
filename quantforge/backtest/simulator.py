import pandas as pd

from quantforge.risk.turnover import calculate_turnover
from quantforge.risk.transaction_cost import apply_transaction_cost


def simulate(
    portfolio_df,
    return_column,
    holding_days,
    round_trip_cost,
):

    portfolio_df = portfolio_df.copy()

    portfolio_df["Date"] = pd.to_datetime(
        portfolio_df["Date"]
    )

    rebalance_dates = sorted(
        portfolio_df["Date"].unique()
    )[::holding_days]

    previous_portfolio = {}

    results = []

    for date in rebalance_dates:

        g = portfolio_df[
            portfolio_df["Date"] == date
        ].copy()

        if g.empty:
            continue

        # -----------------------
        # Build portfolio weights
        # -----------------------

        if "Weight" in g.columns:

            current_portfolio = dict(
                zip(
                    g["Ticker"],
                    g["Weight"]
                )
            )

            gross_return = (
                g["Weight"]
                * g[return_column]
            ).sum()

        else:

            equal_weight = (
                1.0 / len(g)
            )

            current_portfolio = {
                t: equal_weight
                for t in g["Ticker"]
            }

            gross_return = (
                g[return_column]
                .mean()
            )

        turnover = calculate_turnover(
            previous_portfolio,
            current_portfolio
        )

        transaction_cost = (
            turnover
            * round_trip_cost
        )

        net_return = apply_transaction_cost(
            gross_return,
            turnover,
            round_trip_cost
        )

        results.append(
            {
                "Date": date,
                "GrossReturn": gross_return,
                "TransactionCost": transaction_cost,
                "Turnover": turnover,
                "Return": net_return,
            }
        )

        previous_portfolio = current_portfolio

    portfolio = pd.DataFrame(results)

    portfolio["Equity"] = (
        1
        + portfolio["Return"]
    ).cumprod()

    return portfolio