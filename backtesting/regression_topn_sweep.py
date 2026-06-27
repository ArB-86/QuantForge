import pandas as pd
import numpy as np

DATA_FILE = (
    "data/checkpoints/"
    "monthly_walkforward_regression.csv"
)

TOPN_LIST = [
    5,
    10,
    20,
    30,
    50,
    100
]

HOLD_DAYS = 5
COST = 0.002

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)

dates = sorted(
    df["Date"].unique()
)

results = []

for TOP_N in TOPN_LIST:

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
            net_return
        )

    portfolio = pd.Series(
        portfolio_returns
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

    win_rate = (
        portfolio > 0
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
        "TopN",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "WinRate"
    ]
)

print("\n====================")
print("TOP-N ROBUSTNESS")
print("====================")
print(
    results.to_string(
        index=False
    )
)

results.to_csv(
    "data/checkpoints/regression_topn_sweep.csv",
    index=False
)
