# File: experiments/model_comparison.py

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
# MODELS
# =====================================

MODELS = {

    "LightGBM_5D": {

        "file":
            "data/checkpoints/monthly_walkforward_regression.csv",

        "score":
            "PRED_RETURN",

        "target":
            "TARGET_5D_RETURN",

        "holding":
            5

    },

    "LightGBM_10D": {

        "file":
            "data/checkpoints/monthly_walkforward_regression_v13.csv",

        "score":
            "PRED_RETURN",

        "target":
            "TARGET_10D_RETURN",

        "holding":
            10

    },

    "LightGBM_20D": {

        "file":
            "data/checkpoints/monthly_walkforward_regression_v14.csv",

        "score":
            "PRED_RETURN",

        "target":
            "TARGET_20D_RETURN",

        "holding":
            20

    },

    "Ensemble": {

        "file":
            "data/checkpoints/monthly_walkforward_ensemble_v1.csv",

        "score":
            "ENSEMBLE_SCORE",

        "target":
            "TARGET_5D_RETURN",

        "holding":
            5

    }

}

# =====================================
# RUN
# =====================================

results = []

print()
print("========================================")
print("MODEL COMPARISON")
print("========================================")

for name, cfg in MODELS.items():

    print()

    print("Running:", name)

    df = pd.read_csv(
        cfg["file"]
    )

    portfolio = build_inverse_volatility_portfolio(

        df,

        score_column=cfg["score"],

        volatility_column="VOL_20D",

        top_n=20

    )

    bt = simulate(

        portfolio,

        return_column=cfg["target"],

        holding_days=cfg["holding"],

        round_trip_cost=0.002

    )

    stats = evaluate(

        bt,

        holding_days=cfg["holding"]

    )

    row = {

        "Model":
            name,

        "Holding":
            cfg["holding"]

    }

    row.update(stats)

    results.append(row)

# =====================================
# OUTPUT
# =====================================

results = pd.DataFrame(results)

print()
print("========================================")
print(results)
print("========================================")
print()

OUTPUT = (
    "experiments/model_comparison.csv"
)

results.to_csv(

    OUTPUT,

    index=False

)

print("Saved:", OUTPUT)