import lightgbm as lgb

class LightGBMModel:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
