import xgboost as xgb
import joblib
from quantforge.modeling.base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {}
        # Enforce GPU by default for professional workloads
        if 'device' not in self.params:
            self.params['device'] = 'cuda'
        
        # Determine task type (Regression by default for quantitative forecasting)
        self.task = self.params.pop('task', 'regression')
        if self.task == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification.")
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = joblib.load(path)
        return instance
