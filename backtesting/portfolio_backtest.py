import pandas as pd
import numpy as np

TOP_N = 10

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

portfolio_returns = []

for date, group in df.groupby("Date"):

    picks = (
        group
        .sort_values(
            "PROBA_UP",
            ascending=False
        )
        .head(TOP_N)
    )

    ret = picks[
        "TARGET_5D_RETURN"
    ].mean()

    portfolio_returns.append(
        [date, ret]
    )

portfolio = pd.DataFrame(
    portfolio_returns,
    columns=[
        "Date",
        "Return"
    ]
)

portfolio["Equity"] = (
    1 + portfolio["Return"]
).cumprod()

portfolio.to_csv(
    "data/checkpoints/portfolio_results.csv",
    index=False
)

# ====================
# METRICS
# ====================

cagr = (
    portfolio["Equity"].iloc[-1]
    **
    (
        252 / len(portfolio)
    )
    - 1
)

vol = (
    portfolio["Return"].std()
    * np.sqrt(252)
)

sharpe = (
    portfolio["Return"].mean()
    /
    portfolio["Return"].std()
    *
    np.sqrt(252)
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

print("\n====================")
print("PORTFOLIO RESULTS")
print("====================")

print(
    "Days:",
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