import joblib
from catboost import CatBoostRegressor, CatBoostClassifier
from quantforge.modeling.base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {}
        if 'task_type' not in self.params:
            self.params['task_type'] = 'GPU'
            
        self.task = self.params.pop('task', 'regression')
        if self.task == 'regression':
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y, verbose=False)
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
