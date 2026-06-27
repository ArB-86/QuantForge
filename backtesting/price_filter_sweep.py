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

PRICE_FILTERS = [
    0,
    20,
    50,
    100,
    200,
    500
]

results = []

for price_filter in PRICE_FILTERS:

    temp_df = df[
        df["Close"] >= price_filter
    ].copy()

    portfolio_returns = []

    dates = sorted(
        temp_df["Date"].unique()
    )

    for i in range(
        0,
        len(dates),
        HOLD_DAYS
    ):

        date = dates[i]

        group = temp_df[
            temp_df["Date"] == date
        ].copy()

        if len(group) < TOP_N:
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

    results.append([
        price_filter,
        cagr,
        sharpe,
        max_dd,
        win_rate
    ])

results = pd.DataFrame(
    results,
    columns=[
        "PriceFilter",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "WinRate"
    ]
)

results = results.sort_values(
    "Sharpe",
    ascending=False
)

results.to_csv(
    "data/checkpoints/price_filter_results.csv",
    index=False
)

print("\n====================")
print("PRICE FILTER SWEEP")
print("====================")
print(results)
