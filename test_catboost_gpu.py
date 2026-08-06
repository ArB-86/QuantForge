from catboost import CatBoostRegressor
import numpy as np
X = np.random.rand(5000, 20)
y = np.random.rand(5000)
model = CatBoostRegressor(iterations=20, task_type='GPU', verbose=False)
model.fit(X, y)
print('CatBoost GPU training successful')
