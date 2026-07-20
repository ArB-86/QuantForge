from lightgbm import LGBMRegressor
from quantforge.common.utils.system import get_num_threads


class ModelManager:

    def __init__(self, config):
        self.config = config

    def build(self):
        # Prefer num_threads from config; fallback to system
        n_jobs = self.config.get("num_threads", get_num_threads())
        return LGBMRegressor(
            device=self.config.get("device", "gpu"),
            objective="regression",
            metric="rmse",
            learning_rate=self.config["learning_rate"],
            num_leaves=self.config["num_leaves"],
            max_depth=self.config["max_depth"],
            n_estimators=self.config["n_estimators"],
            subsample=self.config["subsample"],
            subsample_freq=self.config["subsample_freq"],
            colsample_bytree=self.config["colsample_bytree"],
            min_child_samples=self.config["min_child_samples"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            random_state=self.config.get("random_state", 42),
            force_col_wise=True,
            max_bin=127,
            verbose=-1,
            n_jobs=n_jobs,
        )
