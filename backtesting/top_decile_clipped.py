import pandas as pd

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

df["TARGET_5D_RETURN_CLIPPED"] = (
    df["TARGET_5D_RETURN"]
    .clip(-0.30, 0.30)
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
    "Top:",
    top["TARGET_5D_RETURN_CLIPPED"].mean()
)

print(
    "Bottom:",
    bottom["TARGET_5D_RETURN_CLIPPED"].mean()
)

print(
    "Spread:",
    top["TARGET_5D_RETURN_CLIPPED"].mean()
    -
    bottom["TARGET_5D_RETURN_CLIPPED"].mean()
)

