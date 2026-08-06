import pandas as pd

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_catboost_v10.csv"
)

top_returns = []
bottom_returns = []

for date, g in df.groupby("Date"):

    g = g.sort_values(
        "PRED_RETURN"
    )

    n = max(
        int(len(g) * 0.1),
        1
    )

    bottom = g.head(n)

    top = g.tail(n)

    top_returns.append(
        top["TARGET_5D_RETURN"].mean()
    )

    bottom_returns.append(
        bottom["TARGET_5D_RETURN"].mean()
    )

print(
    "Top:",
    sum(top_returns)
    /
    len(top_returns)
)

print(
    "Bottom:",
    sum(bottom_returns)
    /
    len(bottom_returns)
)

print(
    "Spread:",
    (
        sum(top_returns)
        /
        len(top_returns)
    )
    -
    (
        sum(bottom_returns)
        /
        len(bottom_returns)
    )
)
