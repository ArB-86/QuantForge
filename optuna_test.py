from quantforge.optimization.optuna_engine import (
    OptunaEngine
)

from quantforge.run import run

import json

cfg = json.load(
    open(
        "configs/lightgbm_regressor.json"
    )
)

runner = run

engine = OptunaEngine(
    cfg,
    runner,
)

print(engine)
