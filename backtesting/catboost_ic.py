import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_catboost.csv"
)

ics = []

for date, g in df.groupby("Date"):

    ic, _ = spearmanr(
        g["PRED_RETURN"],
        g["TARGET_5D_RETURN"]
    )

    ics.append(ic)

ics = pd.Series(ics)

print("\n====================")
print("CATBOOST IC")
print("====================")

print("Mean IC:", ics.mean())
print("Median IC:", ics.median())
print("Positive Days:", (ics > 0).mean())
