import os
from pathlib import Path
OUT = Path(os.environ["QF_EXPERIMENT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

DATA = (
    PROJECT_ROOT.parent
    / "data"
    / "checkpoints"
    / "monthly_walkforward_regression_v15.csv"
)

df = pd.read_csv(DATA)

TARGET = "TARGET_20D_RETURN"
PRED = "PRED_RETURN"

print("\nCorrelation")
print(df[[PRED, TARGET]].corr().iloc[0,1])

print("\nPrediction Summary")
print(df[PRED].describe())

for n in [5,10,20,30,50,100]:

    temp = (
        df
        .groupby("Date")
        .apply(
            lambda x:
            x.nlargest(
                n,
                PRED
            )[TARGET].mean()
        )
    )

    print(
        f"\nTop {n}"
    )

    print(
        temp.describe()
    )

print("\nPrediction Deciles")

df["DECILE"] = pd.qcut(
    df[PRED],
    10,
    labels=False
)

table = (
    df
    .groupby("DECILE")[TARGET]
    .agg(
        ["mean","std","count"]
    )
)

print(table)

table.to_csv(
    OUT / "prediction_deciles.csv"
)

print("\nSaved prediction_deciles.csv")
