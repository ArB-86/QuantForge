import json

from quantforge.optimization.optuna_engine import (
    OptunaEngine,
)

from quantforge.research.runner import (
    ExperimentRunner,
)


config = json.load(
    open(
        "configs/lightgbm_regressor.json"
    )
)

runner = ExperimentRunner()

engine = OptunaEngine(
    config,
    runner,
)

study = engine.optimize(
    n_trials=50,
)

print()

print(study.best_value)

print(study.best_params)
