import pandas as pd
import numpy as np

# ====================
# SETTINGS
# ====================

TOP_LIST = [
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    75,
    100
]

HOLD_DAYS = 5

ROUND_TRIP_COST = 0.002

# ====================
# LOAD DATA
# ====================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v12.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

# ====================
# PORTFOLIO CONSTRUCTION
# ====================

dates = sorted(
    df["Date"].unique()
)

results = []

# ====================
# TOP N SENSITIVITY LOOP
# ====================

for TOP_N in TOP_LIST:

    print(f"\nRunning TOP {TOP_N}")

    portfolio_returns = []

    previous_holdings = set()

    for i in range(
        0,
        len(dates),
        HOLD_DAYS
    ):

        date = dates[i]

        group = df[
            df["Date"] == date
        ]

        picks = (
            group
            .sort_values(
                "PRED_RETURN",
                ascending=False
            )
            .head(TOP_N)
        )

        current_holdings = set(
            picks["Ticker"]
        )

        gross_return = (
            picks[
                "TARGET_5D_RETURN"
            ].mean()
        )

        if len(previous_holdings) == 0:

            turnover = 1.0

        else:

            overlap = len(
                previous_holdings &
                current_holdings
            )

            turnover = (
                TOP_N - overlap
            ) / TOP_N

        transaction_cost = (
            turnover *
            ROUND_TRIP_COST
        )

        net_return = (
            gross_return -
            transaction_cost
        )

        portfolio_returns.append(
            [
                date,
                turnover,
                transaction_cost,
                net_return
            ]
        )

        previous_holdings = current_holdings

    portfolio = pd.DataFrame(
        portfolio_returns,
        columns=[
            "Date",
            "Turnover",
            "TransactionCost",
            "Return"
        ]
    )

    portfolio["Equity"] = (
        1 +
        portfolio["Return"]
    ).cumprod()

    years = (
        len(portfolio)
        * HOLD_DAYS
    ) / 252

    cagr = (
        portfolio["Equity"].iloc[-1]
        **
        (1 / years)
        - 1
    )

    vol = (
        portfolio["Return"].std()
        *
        np.sqrt(
            252 / HOLD_DAYS
        )
    )

    sharpe = (
        portfolio["Return"].mean()
        /
        portfolio["Return"].std()
        *
        np.sqrt(
            252 / HOLD_DAYS
        )
    )

    rolling_max = (
        portfolio["Equity"]
        .cummax()
    )

    drawdown = (
        portfolio["Equity"]
        /
        rolling_max
        - 1
    )

    max_dd = drawdown.min()

    win_rate = (
        portfolio["Return"] > 0
    ).mean()

    results.append({

        "TOP_N": TOP_N,

        "CAGR": cagr,

        "Sharpe": sharpe,

        "MaxDD": max_dd,

        "WinRate": win_rate,

        "Turnover": portfolio[
            "Turnover"
        ].mean(),

        "FinalEquity": portfolio[
            "Equity"
        ].iloc[-1]

    })

# ====================
# SAVE RESULTS
# ====================

results = pd.DataFrame(results)

results.to_csv(
    "data/checkpoints/topn_sensitivity_v1.csv",
    index=False
)

print("\n========================")
print(results.sort_values(
    "Sharpe",
    ascending=False
))
