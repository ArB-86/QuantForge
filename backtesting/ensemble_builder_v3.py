
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data" / "checkpoints"

r5 = pd.read_csv(DATA / "monthly_walkforward_regression_v17.csv")
r10 = pd.read_csv(DATA / "monthly_walkforward_regression_v16.csv")
r20 = pd.read_csv(DATA / "monthly_walkforward_regression_v15.csv")

cols = [
    "Date",
    "Ticker"
]

df = (
    r20
    .merge(
        r10[cols + ["PRED_RETURN"]],
        on=cols,
        suffixes=("_20","_10")
    )
    .merge(
        r5[cols + ["PRED_RETURN"]],
        on=cols
    )
)

df.rename(
    columns={
        "PRED_RETURN":"PRED_RETURN_5"
    },
    inplace=True
)

for c in [
    "PRED_RETURN_20",
    "PRED_RETURN_10",
    "PRED_RETURN_5"
]:
    df[c] = (
        df.groupby("Date")[c]
        .rank(pct=True)
    )

df["ENSEMBLE_SCORE"] = (
      0.50*df["PRED_RETURN_20"]
    + 0.30*df["PRED_RETURN_10"]
    + 0.20*df["PRED_RETURN_5"]
)

df.to_csv(
    DATA / "monthly_walkforward_ensemble_v3.csv",
    index=False
)

print(df.head())
print()
print("Saved:", len(df))
