import pandas as pd
import numpy as np

# ==========================
# BEST CONFIGURATION
# ==========================

TOP_N = 20
PRICE_FILTER = 0
HOLD_DAYS = 5
ROUND_TRIP_COST = 0.002

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

# ==========================
# BUILD PORTFOLIO
# ==========================

portfolio_rows = []

for i in range(
    0,
    len(dates),
    HOLD_DAYS
):

    rebalance_date = dates[i]

    day_df = df[
        df["Date"] == rebalance_date
    ]

    if PRICE_FILTER > 0:

        day_df = day_df[
            day_df["Close"]
            >= PRICE_FILTER
        ]

    if len(day_df) < TOP_N:
        continue

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
        - ROUND_TRIP_COST
    )

    portfolio_rows.append(
        [
            rebalance_date,
            net_return
        ]
    )

portfolio = pd.DataFrame(
    portfolio_rows,
    columns=[
        "Date",
        "Return"
    ]
)

portfolio["Date"] = pd.to_datetime(
    portfolio["Date"]
)

portfolio["Year"] = (
    portfolio["Date"]
    .dt.year
)

# ==========================
# YEARLY METRICS
# ==========================

results = []

for year, grp in portfolio.groupby(
    "Year"
):

    equity = (
        1 + grp["Return"]
    ).cumprod()

    annual_return = (
        equity.iloc[-1]
        - 1
    )

    sharpe = (
        grp["Return"].mean()
        /
        grp["Return"].std()
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
        grp["Return"] > 0
    ).mean()

    results.append(
        [
            year,
            annual_return,
            sharpe,
            max_dd,
            win_rate,
            len(grp)
        ]
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Year",
        "Return",
        "Sharpe",
        "MaxDD",
        "WinRate",
        "Trades"
    ]
)

print("\n==============================")
print("YEARLY PERFORMANCE")
print("==============================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n==============================")
print("SUMMARY")
print("==============================")

print(
    "Positive Years:",
    (
        results_df["Return"] > 0
    ).sum(),
    "/",
    len(results_df)
)

print(
    "Average Annual Return:",
    round(
        results_df["Return"].mean()
        * 100,
        2
    ),
    "%"
)

print(
    "Average Sharpe:",
    round(
        results_df["Sharpe"].mean(),
        2
    )
)

results_df.to_csv(
    "data/checkpoints/yearly_analysis.csv",
    index=False
)

print(
    "\nSaved: data/checkpoints/yearly_analysis.csv"
)
