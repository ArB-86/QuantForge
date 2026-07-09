import json

from quantforge.optimization.optuna_engine import (
    OptunaEngine,
)

from quantforge.run import run


config = json.load(
    open(
        "configs/lightgbm_regressor.json"
    )
)

runner = run

engine = OptunaEngine(
    config,
    runner,
)

study = engine.optimize(
    n_trials=1,
)

print()

print(study.best_value)

print(study.best_params)
