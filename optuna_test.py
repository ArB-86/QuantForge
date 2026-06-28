from quantforge.optimization.optuna_engine import (
    OptunaEngine
)

from quantforge.research.runner import (
    ExperimentRunner
)

import json

cfg = json.load(
    open(
        "configs/lightgbm_regressor.json"
    )
)

runner = ExperimentRunner()

engine = OptunaEngine(
    cfg,
    runner,
)

print(engine)
