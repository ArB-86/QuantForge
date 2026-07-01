from quantforge.config.config import Config

from quantforge.optimization.optuna_engine import (
    OptunaEngine,
)

from quantforge.research.runner import (
    ExperimentRunner,
)


cfg = Config(
    "configs/lightgbm_optuna.json"
)

runner = ExperimentRunner()

engine = OptunaEngine(

    cfg.dict(),

    runner,

)

study = engine.optimize(

    n_trials=20,

)

print()
print("=" * 80)
print("BEST SCORE")
print("=" * 80)
print(study.best_value)

print()
print("=" * 80)
print("BEST PARAMS")
print("=" * 80)
print(study.best_params)