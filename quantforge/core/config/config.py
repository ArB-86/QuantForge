import json

from quantforge.features.registry import get_features


class Config:

    def __init__(self, path):

        cfg = json.load(open(path))

        #
        # Default values for backward compatibility
        #

        cfg.setdefault("features", "v3")
        cfg.setdefault("target", "TARGET_20D_RETURN")
        cfg.setdefault("holding_days", 20)
        cfg.setdefault("top_n", 15)
        cfg.setdefault("transaction_cost", 0.003)
        cfg.setdefault("feature_store", "v3")
        cfg.setdefault("seed", 42)

        if isinstance(cfg["features"], str):
            cfg["features"] = get_features(cfg["features"])

        self.data = cfg

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def dict(self):
        return self.data