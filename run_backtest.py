import pandas as pd

# New import
from quantforge.portfolio.allocator import build_portfolio

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate

df = pd.read_csv(
    "data/checkpoints/monthly_walkforward_ensemble_v2.csv"
)

# Build portfolio with inverse volatility weighting
portfolio = build_portfolio(
    df,
    method="inverse_volatility",
    score_column="ENSEMBLE_SCORE",
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