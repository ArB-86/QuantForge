import pandas as pd

from quantforge.risk.turnover import calculate_turnover
from quantforge.risk.transaction_cost import apply_transaction_cost


def simulate(
    portfolio_df,
    return_column="TARGET_5D_RETURN",
    holding_days=5,
    round_trip_cost=0.002
):

    portfolio_df = portfolio_df.copy()
    portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])

    rebalance_dates = sorted(portfolio_df["Date"].unique())[::holding_days]

    previous_holdings = set()
    results = []

    for date in rebalance_dates:

        g = portfolio_df[
            portfolio_df["Date"] == date
        ]

        if len(g) == 0:
            continue

        current_holdings = set(g["Ticker"])

        turnover = calculate_turnover(
            previous_holdings,
            current_holdings
        )

        # --- CHANGED SECTION ---
        if "Weight" in g.columns:
            gross_return = (
                g["Weight"]
                *
                g[return_column]
            ).sum()
        else:
            gross_return = (
                g[return_column].mean()
            )
        # --- END CHANGED SECTION ---

        net_return = apply_transaction_cost(
            gross_return,
            turnover,
            round_trip_cost
        )

        results.append({
            "Date": date,
            "GrossReturn": gross_return,
            "Turnover": turnover,
            "Return": net_return
        })

        previous_holdings = current_holdings

    portfolio = pd.DataFrame(results)

    portfolio["Equity"] = (
        1 + portfolio["Return"]
    ).cumprod()

    return portfolio