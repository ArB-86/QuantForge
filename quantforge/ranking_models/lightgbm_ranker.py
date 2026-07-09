from __future__ import annotations

import importlib
from typing import Any

import joblib

from quantforge.modeling.base import BaseModel
from quantforge.modeling.registry import register_model


def _normalize_kind(kind: str | None) -> str:

    if kind is None:
        return "regressor"

    normalized = kind.lower()

    if normalized in {"regression", "regressor", "lgbmregressor"}:
        return "regressor"

    if normalized in {"classification", "classifier", "lgbmclassifier"}:
        return "classifier"

    if normalized in {"ranking", "ranker", "lgbmranker"}:
        return "ranker"

    raise ValueError(f"Unsupported LightGBM kind: {kind}")


@register_model("lightgbm")
class LightGBMModel(BaseModel):

    def __init__(self, kind: str = "regressor", **params: Any):

        self.kind = _normalize_kind(kind)
        self.params = dict(params)
        self.model = None

    def _get_lightgbm_classes(self):

        module = importlib.import_module("lightgbm")
        return module.LGBMClassifier, module.LGBMRanker, module.LGBMRegressor

    def _build_model(self):

        LGBMClassifier, LGBMRanker, LGBMRegressor = self._get_lightgbm_classes()

        defaults = {
            "device": "gpu",
            "force_col_wise": True,
            "verbose": -1,
        }

        if self.kind == "regressor":
            defaults.update(
                {
                    "objective": "regression",
                    "metric": "rmse",
                }
            )
            model_cls = LGBMRegressor

        elif self.kind == "classifier":
            defaults.update(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                }
            )
            model_cls = LGBMClassifier

        else:
            defaults.update(
                {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                }
            )
            model_cls = LGBMRanker

        defaults.update(self.params)
        return model_cls(**defaults)

    def _ensure_model(self):

        if self.model is None:
            self.model = self._build_model()

        return self.model

    def fit(self, X, y, group=None):

        model = self._ensure_model()

        if self.kind == "ranker" and group is not None:
            model.fit(X, y, group=group)
        else:
            model.fit(X, y)

        return self

    def predict(self, X):

        return self._ensure_model().predict(X)

    def predict_proba(self, X):

        model = self._ensure_model()

        if not hasattr(model, "predict_proba"):
            raise NotImplementedError

        return model.predict_proba(X)

    def save(self, path):

        joblib.dump(self, path)

    @classmethod
    def load(cls, path):

        return joblib.load(path)

    def __getattr__(self, name):

        model = self._ensure_model()
        return getattr(model, name)