import pandas as pd
import numpy as np

# ==========================
# CONFIGS TO TEST
# ==========================

PRICE_FILTERS = [
    0,
    50,
    100,
    200
]

TOP_NS = [
    10,
    20,
    30,
    50
]

COSTS = [
    0.002,   # 20 bps
    0.005,   # 50 bps
    0.010    # 100 bps
]

HOLD_DAYS = 5

# ==========================
# LOAD DATA
# ==========================

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

# ==========================
# LOOP
# ==========================

for price_filter in PRICE_FILTERS:

    for top_n in TOP_NS:

        for cost in COSTS:

            portfolio_returns = []

            for i in range(
                0,
                len(dates),
                HOLD_DAYS
            ):

                rebalance_date = dates[i]

                day_df = df[
                    df["Date"] == rebalance_date
                ]

                if price_filter > 0:

                    day_df = day_df[
                        day_df["Close"]
                        >= price_filter
                    ]

                if len(day_df) < top_n:
                    continue

                picks = (
                    day_df
                    .sort_values(
                        "PROBA_UP",
                        ascending=False
                    )
                    .head(top_n)
                )

                gross_return = (
                    picks[
                        "TARGET_5D_RETURN"
                    ].mean()
                )

                net_return = (
                    gross_return
                    - cost
                )

                portfolio_returns.append(
                    net_return
                )

            if len(portfolio_returns) == 0:
                continue

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
                    price_filter,
                    top_n,
                    cost,
                    cagr,
                    sharpe,
                    max_dd,
                    win_rate
                ]
            )

# ==========================
# SAVE
# ==========================

results_df = pd.DataFrame(
    results,
    columns=[
        "PriceFilter",
        "TopN",
        "Cost",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "WinRate"
    ]
)

results_df = (
    results_df
    .sort_values(
        "Sharpe",
        ascending=False
    )
)

results_df.to_csv(
    "data/checkpoints/robustness_results.csv",
    index=False
)

print("\n==========================")
print("TOP 20 CONFIGURATIONS")
print("==========================")

print(
    results_df.head(20)
)

print("\nSaved:")
print(
    "data/checkpoints/robustness_results.csv"
)

