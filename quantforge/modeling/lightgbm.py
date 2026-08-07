import joblib
import lightgbm as lgb
from quantforge.modeling.base import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {}
        self.task = self.params.pop('task', 'regression')
        if self.task == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
