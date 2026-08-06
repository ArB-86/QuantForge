# File: experiments/holding_period_sweep.py

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

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v1.csv"
)

portfolio = build_inverse_volatility_portfolio(
    df,
    score_column="ENSEMBLE_SCORE",
    volatility_column="VOL_20D",
    top_n=20
)

# ======================================
# HOLDING PERIODS
# ======================================

holding_periods = [
    1,
    3,
    5,
    10,
    15,
    20
]

results = []

for holding in holding_periods:

    print(f"Running Holding = {holding}")

    bt = simulate(
        portfolio,
        return_column="TARGET_5D_RETURN",
        holding_days=holding,
        round_trip_cost=0.002
    )

    stats = evaluate(
        bt,
        holding_days=holding
    )

    stats["HoldingDays"] = holding

    results.append(stats)

results = pd.DataFrame(results)

cols = [
    "HoldingDays",
    "Final Equity",
    "CAGR",
    "Sharpe",
    "Volatility",
    "Max Drawdown",
    "Win Rate"
]

results = results[cols]

print()
print(results)

results.to_csv(
    "experiments/holding_period_results.csv",
    index=False
)

print()
print("Saved -> experiments/holding_period_results.csv")
