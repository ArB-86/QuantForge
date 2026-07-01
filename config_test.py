from quantforge.config.config import Config

cfg = Config(
    "configs/lightgbm_regressor.json"
)

print(type(cfg))

print(len(cfg["features"]))

print(cfg["target"])
