import pandas as pd

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

df = df.sort_values(
    "PROBA_UP",
    ascending=False
)

top = df.head(
    int(len(df) * 0.10)
)

bottom = df.tail(
    int(len(df) * 0.10)
)

print(
    "Top 10% Avg 5D Return:",
    top["TARGET_5D_RETURN"].mean()
)

print(
    "Bottom 10% Avg 5D Return:",
    bottom["TARGET_5D_RETURN"].mean()
)

print(
    "Spread:",
    top["TARGET_5D_RETURN"].mean()
    -
    bottom["TARGET_5D_RETURN"].mean()
)
