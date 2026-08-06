import pandas as pd
import numpy as np

# ====================
# SETTINGS
# ====================

TOP_LIST = [3,5,10,15,20,30,50]
HOLD_DAYS = 5
COST = 0.002

# ====================
# LOAD DATA
# ====================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v8.csv"
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

for TOP_N in TOP_LIST:

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

        picks = (
            group
            .sort_values(
                "PRED_RETURN",
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
            - COST
        )

        portfolio_returns.append(
            [
                date,
                net_return
            ]
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

    results.append(
        [
            TOP_N,
            cagr,
            sharpe,
            max_dd,
            win_rate
        ]
    )

results = pd.DataFrame(
    results,
    columns=[
        "TOP_N",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "WinRate"
    ]
)

print("\n====================")
print("TOP N SWEEP")
print("====================")
print(
    results.sort_values(
        "Sharpe",
        ascending=False
    )
)
# ====================
# PORTFOLIO DF
# ====================

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
    "data/checkpoints/portfolio_results_regression.csv",
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
print("REGRESSION PORTFOLIO")
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
