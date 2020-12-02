# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐수를 줄인다.
# 3. regularization

from sklearn.datasets import load_boston
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
max_depth         = 5
learning_rate     = 1
n_estimators      = 300
n_jobs            = -1
colsample_bylevel = 1
colsample_bytree  = 1

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                     n_jobs=n_jobs, colsample_bylevel=colsample_bylevel,
                     colsample_bytree=colsample_bytree)

boston = load_boston()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.7, random_state=66,shuffle=True)

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc1 : ",acc)

plot_importance(model)
plt.show()


model = XGBRegressor()

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc2 : ",acc)

plot_importance(model)
plt.show()

# score 디폴트로 했던놈과 성능비교
# acc1 :  0.8470606908053899
# acc2 :  0.9024460463247372




