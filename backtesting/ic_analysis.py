import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_predictions.csv"
)

daily_ic = []

for date, g in df.groupby("Date"):

    if len(g) < 20:
        continue

    ic, _ = spearmanr(
        g["PROBA_UP"],
        g["TARGET_5D_RETURN"]
    )

    daily_ic.append(ic)

daily_ic = pd.Series(daily_ic)

print("\n====================")
print("IC ANALYSIS")
print("====================")

print(
    "Mean IC:",
    daily_ic.mean()
)

print(
    "Median IC:",
    daily_ic.median()
)

print(
    "Positive IC Days:",
    (daily_ic > 0).mean()
)

print(
    "IC Std:",
    daily_ic.std()
)

print(
    "Information Ratio:",
    daily_ic.mean()
    /
    daily_ic.std()
)
