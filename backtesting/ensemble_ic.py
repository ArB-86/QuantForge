import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble.csv"
)

ics = []

for date, g in df.groupby("Date"):

    ic, _ = spearmanr(
        g["ENSEMBLE_PRED"],
        g["TARGET_5D_RETURN"]
    )

    ics.append(ic)

ics = pd.Series(ics)

print("\n====================")
print("ENSEMBLE IC")
print("====================")

print(
    "Mean IC:",
    ics.mean()
)

print(
    "Median IC:",
    ics.median()
)

print(
    "Positive Days:",
    (ics > 0).mean()
)
