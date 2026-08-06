import pandas as pd

from quantforge.portfolio.inverse_volatility import (
    build_inverse_volatility_portfolio
)

from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate

FILES = {

    "v14":
        "data/checkpoints/monthly_walkforward_ensemble_v1.csv",

    "v15":
        "data/checkpoints/monthly_walkforward_ensemble_v2.csv"

}

rows = []

for name, file in FILES.items():

    print("Running", name)

    df = pd.read_csv(file)

    portfolio = build_inverse_volatility_portfolio(

        df,

        score_column="ENSEMBLE_SCORE",

        volatility_column="VOL_20D",

        top_n=20

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

    rows.append({

        "Version": name,

        **stats

    })

results = pd.DataFrame(rows)

print()
print(results)

results.to_csv(

    "experiments/v14_vs_v15.csv",

    index=False

)

print()

print("Saved: experiments/v14_vs_v15.csv")
