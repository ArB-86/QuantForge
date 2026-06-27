import pandas as pd
import numpy as np

# ====================
# SETTINGS
# ====================

TOP_N = 10
HOLD_DAYS = 5
# ROUND_TRIP_COST used for turnover-based transaction cost
ROUND_TRIP_COST = 0.002

# ====================
# LOAD DATA
# ====================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v13.csv"
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

# Step 8: initialize separate previous sets for long and short
previous_long = set()
previous_short = set()

portfolio_returns = []

for i in range(
    0,
    len(dates),
    HOLD_DAYS
):

    date = dates[i]

    group = df[
        df["Date"] == date
    ]

    # Step 4: produce long and short picks
    long_picks = (
        group
        .sort_values(
            "PRED_RETURN",
            ascending=False
        )
        .head(TOP_N)
    )

    short_picks = (
        group
        .sort_values(
            "PRED_RETURN",
            ascending=True
        )
        .head(TOP_N)
    )

    # Step 5: current long/short sets
    current_long = set(
        long_picks["Ticker"]
    )

    current_short = set(
        short_picks["Ticker"]
    )

    # Step 6: separate long/short returns
    long_return = (
        long_picks[
            "TARGET_5D_RETURN"
        ].mean()
    )

    short_return = (
        short_picks[
            "TARGET_5D_RETURN"
        ].mean()
    )

    # Step 7: turnover calculation using separate overlaps
    if len(previous_long) == 0 and len(previous_short) == 0:
        turnover = 1.0
    else:
        overlap_long = len(
            previous_long &
            current_long
        )

        overlap_short = len(
            previous_short &
            current_short
        )

        turnover_long = (
            TOP_N - overlap_long
        ) / TOP_N

        turnover_short = (
            TOP_N - overlap_short
        ) / TOP_N

        turnover = (
            turnover_long +
            turnover_short
        ) / 2

    # Step 9: transaction cost and long/short net returns
    transaction_cost = (
        turnover *
        ROUND_TRIP_COST
    )

    net_long = (
        long_return -
        transaction_cost
    )

    net_short = (
        short_return +
        transaction_cost
    )

    long_short_return = (
        net_long -
        net_short
    )

    # Step 10: append with new structure
    portfolio_returns.append(
        [
            date,
            long_return,
            short_return,
            transaction_cost,
            turnover,
            long_short_return
        ]
    )

    # Step 11: update previous_long / previous_short
    previous_long = current_long
    previous_short = current_short

# ====================
# PORTFOLIO DF (updated columns)
# ====================

portfolio = pd.DataFrame(
    portfolio_returns,
    columns=[
        "Date",
        "LongReturn",
        "ShortReturn",
        "TransactionCost",
        "Turnover",
        "Return"
    ]
)

# Return now represents the long-short return
portfolio["Equity"] = (
    1 + portfolio["Return"]
).cumprod()

# Step 13: save to separate CSV
portfolio.to_csv(
    "data/checkpoints/long_short_results_v2.csv",
    index=False
)

# ====================
# METRICS (updated to use long-short Return)
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
print("LONG-SHORT PORTFOLIO")
print("====================")

print(
    "Rebalances:",
    len(portfolio)
)

print(
    "TOP N:",
    TOP_N
)

print(
    "Portfolio:",
    "Long Top 10 / Short Bottom 10"
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
