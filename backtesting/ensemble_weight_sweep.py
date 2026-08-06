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

weights = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9)
]

results = []

for lgb_w, cat_w in weights:

    temp_df = df.copy()

    temp_df["COMBINED_PRED"] = (
        lgb_w * temp_df["PRED_RETURN"]
        +
        cat_w * temp_df["CATBOOST_PRED"]
    )

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

        picks = (
            group
            .sort_values(
                "COMBINED_PRED",
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
        lgb_w,
        cat_w,
        cagr,
        sharpe,
        max_dd,
        win_rate
    ])

results = pd.DataFrame(
    results,
    columns=[
        "LGBM",
        "CATBOOST",
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
    "data/checkpoints/ensemble_weight_results.csv",
    index=False
)

print("\n====================")
print("ENSEMBLE WEIGHT SWEEP")
print("====================")

print(results)
