import pandas as pd
import numpy as np

TOP_N = 20
HOLD_DAYS = 5

# 10 bps entry + 10 bps exit
ROUND_TRIP_COST = 0.002

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

dates = sorted(
    df["Date"].unique()
)

results = []

for i in range(
    0,
    len(dates),
    HOLD_DAYS
):

    rebalance_date = dates[i]

    day_df = df[
        df["Date"] == rebalance_date
    ]

    picks = (
        day_df
        .sort_values(
            "PROBA_UP",
            ascending=False
        )
        .head(TOP_N)
    )

    gross_return = (
        picks[
            "TARGET_5D_RETURN"
        ].mean()
    )

    net_return = (
        gross_return
        -
        ROUND_TRIP_COST
    )

    results.append(
        [
            rebalance_date,
            gross_return,
            net_return
        ]
    )

portfolio = pd.DataFrame(
    results,
    columns=[
        "Date",
        "GrossReturn",
        "NetReturn"
    ]
)

portfolio["Equity"] = (
    1 + portfolio["NetReturn"]
).cumprod()

# =====================
# Metrics
# =====================

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
    portfolio["NetReturn"].std()
    *
    np.sqrt(
        252 / HOLD_DAYS
    )
)

sharpe = (
    portfolio["NetReturn"].mean()
    /
    portfolio["NetReturn"].std()
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
    portfolio["NetReturn"] > 0
).mean()

print("\n====================")
print("PORTFOLIO V2")
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

portfolio.to_csv(
    "data/checkpoints/portfolio_v2.csv",
    index=False
)
