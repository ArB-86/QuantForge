# File: experiments/topn_sweep.py

import os
import sys
import pandas as pd

# =====================================
# PROJECT ROOT
# =====================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================
# IMPORTS
# =====================================

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio
)

from quantforge.backtest.simulator import (
    simulate
)

from quantforge.backtest.metrics import (
    evaluate
)

# =====================================
# LOAD DATA
# =====================================

DATA_FILE = (
    "data/checkpoints/"
    "monthly_walkforward_ensemble_v1.csv"
)

df = pd.read_csv(DATA_FILE)

print()
print("==============================")
print("Top-N Robustness Sweep")
print("==============================")
print()

results = []

# =====================================
# SWEEP
# =====================================

for top_n in [5, 10, 15, 20, 25, 30]:

    print(f"Running Top {top_n}")

    portfolio = build_inverse_volatility_portfolio(

        df,

        score_column="ENSEMBLE_SCORE",

        volatility_column="VOL_20D",

        top_n=top_n

    )

    bt = simulate(

        portfolio,

        return_column="TARGET_5D_RETURN",

        holding_days=5,

        round_trip_cost=0.002

    )

    stats = evaluate(

        bt,

        holding_days=5

    )

    results.append({

        "TopN": top_n,

        "Final Equity": stats["Final Equity"],

        "CAGR": stats["CAGR"],

        "Sharpe": stats["Sharpe"],

        "Volatility": stats["Volatility"],

        "Max Drawdown": stats["Max Drawdown"],

        "Win Rate": stats["Win Rate"]

    })

# =====================================
# RESULTS
# =====================================

results = pd.DataFrame(results)

print()
print("==============================")
print(results)
print("==============================")
print()

OUTPUT_FILE = (
    "experiments/"
    "topn_sweep_results.csv"
)

results.to_csv(

    OUTPUT_FILE,

    index=False

)

print("Saved:", OUTPUT_FILE)
