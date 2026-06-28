import os
from pathlib import Path
OUT = Path(os.environ["QF_EXPERIMENT_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantforge.engine.manager import (
    ExperimentManager
)

import numpy as np
import pandas as pd

config = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "configs/lgbm20.json"
)

manager = ExperimentManager(config)

exp_id, folder = manager.create()

print()

print("="*80)

print("Experiment")

print(exp_id)

print(folder)

print("="*80)

DATA = (
    PROJECT_ROOT.parent
    / "data"
    / "checkpoints"
    / "monthly_walkforward_regression_v15.csv"
)

df = pd.read_csv(DATA)
df["Date"] = pd.to_datetime(df["Date"])

TOPS = [5,10,15,20,25,30,40,50,75,100]
ROUND_TRIP_COST = 0.003

results = []

for TOP_N in TOPS:

    returns = []

    dates = sorted(df["Date"].unique())
    dates = dates[::20]

    for d in dates:

        temp = df[df["Date"] == d]

        temp = temp[
            (temp["BULL_REGIME"] == 1) &
            (temp["HIGH_VOL_REGIME"] == 0)
        ]

        picks = (
            temp
            .sort_values(
                "PRED_RETURN",
                ascending=False
            )
            .head(TOP_N)
        )

        if len(picks) == 0:
            continue

        portfolio_return = picks["TARGET_20D_RETURN"].mean()
        portfolio_return -= ROUND_TRIP_COST

        returns.append(portfolio_return)

    r = pd.Series(returns)

    equity = (1+r).cumprod()

    years = len(r)*20/252

    cagr = equity.iloc[-1]**(1/years)-1

    sharpe = (
        r.mean()
        /
        r.std()
        *
        np.sqrt(252/20)
    )

    rolling = equity.cummax()

    mdd = (
        equity/rolling-1
    ).min()

    win = (r>0).mean()

    results.append([
        TOP_N,
        cagr,
        sharpe,
        mdd,
        win,
        equity.iloc[-1]
    ])

table = pd.DataFrame(
    results,
    columns=[
        "TopN",
        "CAGR",
        "Sharpe",
        "MaxDD",
        "WinRate",
        "FinalEquity"
    ]
)

table = table.sort_values(
    "Sharpe",
    ascending=False
)

print()
print(table)

Path("results").mkdir(
    exist_ok=True
)

table.to_csv(
    OUT / "topn_sweep.csv",
    index=False
)

print("\nSaved results/topn_sweep.csv")