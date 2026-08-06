import xgboost as xgb
import numpy as np
X = np.random.rand(5000, 50)
y = np.random.rand(5000)
model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=20)
model.fit(X, y)
print('XGBoost GPU training successful')
