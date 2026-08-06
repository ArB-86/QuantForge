import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_regression_v12.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year

results = []

for year in sorted(df["Year"].unique()):

    temp = df[
        df["Year"] == year
    ]

    daily_ic = []

    for _, day in temp.groupby("Date"):

        if len(day) < 5:
            continue

        ic = spearmanr(
            day["PRED_RETURN"],
            day["TARGET_5D_RETURN"]
        )[0]

        if pd.notna(ic):
            daily_ic.append(ic)

    if len(daily_ic) == 0:
        continue

    results.append({

        "Year": year,

        "MeanIC": np.mean(daily_ic),

        "MedianIC": np.median(daily_ic),

        "StdIC": np.std(daily_ic),

        "PositiveDays": (
            np.array(daily_ic) > 0
        ).mean(),

        "TradingDays": len(daily_ic)

    })

results = pd.DataFrame(results)

print("\n========================")
print("YEAR-WISE IC")
print("========================")
print(results)

results.to_csv(
    "data/checkpoints/yearly_ic_analysis_v1.csv",
    index=False
)