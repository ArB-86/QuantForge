# File: experiments/transaction_cost_sweep.py

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio
)

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v1.csv"
)

portfolio = build_inverse_volatility_portfolio(

    df,

    score_column="ENSEMBLE_SCORE",

    volatility_column="VOL_20D",

    top_n=20

)

# =====================================
# COST SWEEP
# =====================================

costs = [
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.010
]

results = []

for cost in costs:

    print(f"Running Cost = {cost:.4f}")

    bt = simulate(

        portfolio,

        return_column="TARGET_5D_RETURN",

        holding_days=5,

        round_trip_cost=cost

    )

    stats = evaluate(

        bt,

        holding_days=5

    )

    results.append({

        "Cost": cost,

        **stats

    })

results = pd.DataFrame(results)

print()
print(results)

results.to_csv(

    "experiments/transaction_cost_sweep.csv",

    index=False

)

print()

print("Saved -> experiments/transaction_cost_sweep.csv")
