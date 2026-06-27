import pandas as pd
import numpy as np

TOP_N = 20
BOTTOM_N = 20

HOLD_DAYS = 5
COST = 0.002

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

dates = sorted(
    df["Date"].unique()
)

returns = []

for i in range(
    0,
    len(dates),
    HOLD_DAYS
):

    date = dates[i]

    day_df = df[
        df["Date"] == date
    ]

    longs = day_df[
        day_df["Close"]
        >
        day_df["EMA200"]
    ]

    shorts = day_df[
        day_df["Close"]
        <
        day_df["EMA200"]
    ]

    if len(longs) < TOP_N:
        continue

    if len(shorts) < BOTTOM_N:
        continue

    long_port = (
        longs
        .sort_values(
            "PROBA_UP",
            ascending=False
        )
        .head(TOP_N)
    )

    short_port = (
        shorts
        .sort_values(
            "PROBA_UP",
            ascending=True
        )
        .head(BOTTOM_N)
    )

    long_ret = (
        long_port[
            "TARGET_5D_RETURN"
        ].mean()
    )

    short_ret = (
        short_port[
            "TARGET_5D_RETURN"
        ].mean()
    )

    net_ret = (
        long_ret
        -
        short_ret
        -
        COST
    )

    returns.append(
        net_ret
    )

portfolio = pd.Series(
    returns
)

equity = (
    1 + portfolio
).cumprod()

years = (
    len(portfolio)
    * HOLD_DAYS
) / 252

cagr = (
    equity.iloc[-1]
    **
    (1 / years)
    - 1
)

sharpe = (
    portfolio.mean()
    /
    portfolio.std()
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

print("\n====================")
print("TREND FILTERED")
print("====================")

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
        (portfolio > 0).mean() * 100,
        2
    ),
    "%"
)
