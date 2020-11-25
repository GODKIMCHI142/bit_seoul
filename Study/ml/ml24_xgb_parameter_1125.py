# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from xgboost import XGBClassifier, XGBRegressor, plot_importance


model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                     n_jobs=n_jobs, colsample_bylevel=colsample_bylevel,
                     colsample_bytree=colsample_bytree)