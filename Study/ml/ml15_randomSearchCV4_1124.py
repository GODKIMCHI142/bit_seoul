# diabetes data
# 모델 : RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {"n_estimators"      : [100, 200]},
    {"max_depth"         : [6, 8, 10, 12]},
    {"min_samples_leaf"  : [3, 5, 7, 10]},
    {"min_samples_split" : [2, 3, 5, 10]},
    {"n_jobs"            : [-1]}
]

parameters2 = [
    {"n_estimators"      : [101, 200],
     "max_depth"         : [6, 8, 10, 12],
     "min_samples_leaf"  : [3, 5, 7, 10],
     "min_samples_split" : [2, 3, 5, 10],
     "n_jobs"            : [-1]}
]

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66,shuffle=True)

# 2. 모델
kfold = KFold(n_splits=5,shuffle=True)
# model = SVC()
model = RandomizedSearchCV(RandomForestRegressor(), parameters2, cv=kfold, verbose=1)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print("최적의 모델    : ",model.best_estimator_)
print("최적의 매개변수 : ",model.best_params_)

y_predict = model.predict(x_test)
# print("최종 정답률     : ",accuracy_score(y_test,y_predict))

from sklearn.metrics import r2_score
print("최종 정답률     : ",r2_score(y_test,y_predict))





# 최적의 모델    :  RandomForestRegressor(max_depth=12, min_samples_leaf=3, min_samples_split=10,
#                       n_estimators=101, n_jobs=-1)
# 최적의 매개변수 :  {'n_jobs': -1, 'n_estimators': 101, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_depth': 12}
# 최종 정답률     :  0.4466715937791472