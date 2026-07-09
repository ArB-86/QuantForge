from __future__ import annotations

import importlib
import json
from typing import Any

from quantforge.modeling.registry import MODEL_REGISTRY, get_model


class ModelFactory:

    @staticmethod
    def create(name: str, **kwargs: Any):

        normalized_name = name.lower()

        if normalized_name in {"lightgbm_regressor", "lightgbm_classifier", "lightgbm_ranker"}:
            kwargs = dict(kwargs)
            if normalized_name.endswith("ranker"):
                kwargs.setdefault("kind", "ranker")
            elif normalized_name.endswith("classifier"):
                kwargs.setdefault("kind", "classifier")
            else:
                kwargs.setdefault("kind", "regressor")
            normalized_name = "lightgbm"

        if normalized_name == "lightgbm" and normalized_name not in MODEL_REGISTRY:
            importlib.import_module("quantforge.modeling.lightgbm")

        model_cls = get_model(normalized_name)
        return model_cls(**kwargs)


def build(config):

    if isinstance(config, str):

        with open(config) as fp:

            config = json.load(fp)

    model_name = config.get("model", "lightgbm")

    model_cfg = {
        key: value
        for key, value in config.items()
        if key not in {
            "name",
            "model",
            "target",
            "data_path",
            "checkpoint_file",
            "model_file",
            "prediction_file",
            "feature_store",
            "top_n",
            "portfolio",
            "max_stock_weight",
            "target_volatility",
            "holding_days",
            "transaction_cost",
            "seed",
            "features",
        }
    }

    if model_name == "lightgbm_ranker":
        return ModelFactory.create("lightgbm", kind="ranker", **model_cfg)

    if model_name == "lightgbm_classifier":
        return ModelFactory.create("lightgbm", kind="classifier", **model_cfg)

    if model_name == "lightgbm":
        return ModelFactory.create("lightgbm", kind="regressor", **model_cfg)

    return ModelFactory.create(model_name, **model_cfg)