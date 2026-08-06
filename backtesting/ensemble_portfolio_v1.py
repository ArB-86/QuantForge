import pandas as pd
import numpy as np

# ====================
# SETTINGS
# ====================

TOP_N = 20
HOLD_DAYS = 5
# ROUND_TRIP_COST used for turnover-based transaction cost
ROUND_TRIP_COST = 0.002

# ====================
# LOAD DATA
# ====================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v1.csv"
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

# ====================
# REPLACED PORTFOLIO LOGIC (turnover + transaction cost)
# ====================

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
            "ENSEMBLE_SCORE",
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
            gross_return,
            transaction_cost,
            turnover,
            net_return
        ]
    )

    previous_holdings = current_holdings

# ====================
# PORTFOLIO DF (updated columns)
# ====================

portfolio = pd.DataFrame(
    portfolio_returns,
    columns=[
        "Date",
        "GrossReturn",
        "TransactionCost",
        "Turnover",
        "Return"
    ]
)

portfolio["Equity"] = (
    1 + portfolio["Return"]
).cumprod()

portfolio.to_csv(
    "data/checkpoints/portfolio_results_regression_v14.csv",
    index=False
)

# ====================
# METRICS
# ====================

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

# ====================
# OUTPUT
# ====================

print("\n====================")
print("REGRESSION PORTFOLIO (with turnover & transaction cost)")
print("====================")

print(
    "Rebalances:",
    len(portfolio)
)

print(
    "Final Equity:",
    round(
        portfolio["Equity"].iloc[-1],
        4
    )
)

print(
    "CAGR:",
    round(
        cagr * 100,
        2
    ),
    "%"
)

print(
    "Sharpe:",
    round(
        sharpe,
        2
    )
)

print(
    "Volatility:",
    round(
        vol * 100,
        2
    ),
    "%"
)

print(
    "Max Drawdown:",
    round(
        max_dd * 100,
        2
    ),
    "%"
)

print(
    "Win Rate:",
    round(
        win_rate * 100,
        2
    ),
    "%"
)
