import pandas as pd
import numpy as np

TOP_N = 10
HOLD_DAYS = 5
COST = 0.002

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

portfolio_returns = []

dates = sorted(
    df["Date"].unique()
)

for i in range(
    0,
    len(dates),
    HOLD_DAYS
):

    date = dates[i]

    group = df[
        df["Date"] == date
    ].copy()

    if len(group) < TOP_N:
        continue

    # =====================
    # REGIME FILTER
    # =====================

    regime = (
        group["HIGH_VOL_REGIME"]
        .iloc[0]
    )

    if regime == 0:

        portfolio_returns.append(
            0.0
        )

        continue

    picks = (
        group
        .sort_values(
            "ENSEMBLE_PRED",
            ascending=False
        )
        .head(TOP_N)
        .copy()
    )

    picks["VOL_20D"] = (
        picks["VOL_20D"]
        .abs()
        .clip(lower=0.0001)
    )

    picks["INV_VOL"] = (
        1.0 /
        picks["VOL_20D"]
    )

    picks["WEIGHT"] = (
        picks["INV_VOL"]
        /
        picks["INV_VOL"].sum()
    )

    gross_return = (
        picks["WEIGHT"]
        *
        picks["TARGET_5D_RETURN"]
    ).sum()

    net_return = (
        gross_return
        - COST
    )

    portfolio_returns.append(
        net_return
    )

portfolio_returns = pd.Series(
    portfolio_returns
)

equity = (
    1 + portfolio_returns
).cumprod()

years = (
    len(portfolio_returns)
    * HOLD_DAYS
) / 252

cagr = (
    equity.iloc[-1]
    **
    (1 / years)
    - 1
)

sharpe = (
    portfolio_returns.mean()
    /
    portfolio_returns.std()
    *
    np.sqrt(
        252 / HOLD_DAYS
    )
)

rolling_max = (
    equity.cummax()
)

drawdown = (
    equity
    /
    rolling_max
    - 1
)

max_dd = (
    drawdown.min()
)

win_rate = (
    portfolio_returns > 0
).mean()

print("\n====================")
print("REGIME SWITCH PORTFOLIO")
print("====================")

print(
    "Rebalances:",
    len(portfolio_returns)
)

print(
    "Final Equity:",
    round(
        equity.iloc[-1],
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
    "MaxDD:",
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
