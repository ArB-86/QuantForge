import pandas as pd

# Old import (remove)
# from quantforge.portfolio.equal_weight import build_equal_weight_portfolio

# New import
from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio
)

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v2.csv"
)

# Build portfolio with inverse volatility weighting
portfolio = build_inverse_volatility_portfolio(
    df,
    score_column="ENSEMBLE_SCORE",     # correct column name
    volatility_column="VOL_20D",
    top_n=10
)

results = simulate(
    portfolio,
    return_column="TARGET_5D_RETURN",
    holding_days=5,
    round_trip_cost=0.002
)

stats = evaluate(
    results,
    holding_days=5
)

print()
print("====================")
for k, v in stats.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(k, v)