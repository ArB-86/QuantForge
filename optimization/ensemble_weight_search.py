import os
import sys
import itertools
import pandas as pd

# ==========================================
# PROJECT ROOT
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================
# IMPORTS
# ==========================================

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio
)

from quantforge.backtest.simulator import simulate

from quantforge.backtest.metrics import evaluate

# ==========================================
# LOAD DATA
# ==========================================

print("Loading ensemble...")

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v2.csv"
)

results = []

weights = [i / 10 for i in range(11)]

# ==========================================
# GRID SEARCH
# ==========================================

for w5, w10, w20 in itertools.product(weights, weights, weights):

    if round(w5 + w10 + w20, 5) != 1.0:
        continue

    temp = df.copy()

    temp["ENSEMBLE_SCORE"] = (
        w5 * temp["RANK5"] +
        w10 * temp["RANK10"] +
        w20 * temp["RANK20"]
    )

    portfolio = build_inverse_volatility_portfolio(
        temp,
        score_column="ENSEMBLE_SCORE",
        volatility_column="VOL_20D",
        top_n=10
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
        "W5": w5,
        "W10": w10,
        "W20": w20,
        **stats
    })

results = pd.DataFrame(results)

results = results.sort_values(
    "Sharpe",
    ascending=False
)

os.makedirs("optimization", exist_ok=True)

results.to_csv(
    "optimization/ensemble_weight_results.csv",
    index=False
)

print()
print("=" * 60)
print(results.head(20))
print("=" * 60)
print()
print("Saved: optimization/ensemble_weight_results.csv")