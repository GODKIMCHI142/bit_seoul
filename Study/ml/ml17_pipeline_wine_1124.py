# wine data
# 모델 : RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
# from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {"svm__n_estimators"      : [100, 200]},
    {"svm__max_depth"         : [6, 8, 10, 12]},
    {"svm__min_samples_leaf"  : [3, 5, 7, 10]},
    {"svm__min_samples_split" : [2, 3, 5, 10]},
    {"svm__n_jobs"            : [-1]}
]

parameters2 = [
    {"svm__n_estimators"      : [50, 100, 150, 200],
     "svm__max_depth"         : [4, 6, 8, 10, 12],
     "svm__min_samples_leaf"  : [3, 5, 7, 10, 15, 20],
     "svm__min_samples_split" : [2, 3, 5, 10, 15, 20],
     "svm__max_leaf_nodes"    : [40, 50, 60, 70, 80, 90, 100],
     "svm__n_jobs"            : [-1]}
]

wine = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")

x = wine.drop("quality", axis=1)
x = np.array(x)
y = wine["quality"]


print(x.shape) # (4898, 11)
print(y.shape) # (4898,)

# newlist = []
# for i in list(y):
#     if   i <= 4:
#         newlist += [0]
#     elif i <= 7:
#         newlist += [1]
#     else: 
#         newlist += [2]

# y = newlist
# y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66,shuffle=True)

# 2. 모델
from sklearn.pipeline import Pipeline, make_pipeline
 
pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",RandomForestClassifier())])

model = RandomizedSearchCV(pipe, parameters, cv=6, verbose=2)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측


print("최적의 모델    : ",model.best_estimator_)
print("최적의 매개변수 : ",model.best_params_)


y_predict = model.predict(x_test)
print("최종 정답률     : ",accuracy_score(y_test,y_predict))

from sklearn.metrics import r2_score
# print("최종 정답률     : ",r2_score(y_test,y_predict))


# GridSearchCV
# [Parallel(n_jobs=1)]: Done 25200 out of 25200 | elapsed: 52.9min finished
# 최적의 모델    :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('svm',
#                  RandomForestClassifier(max_depth=12, max_leaf_nodes=100,
#                                         min_samples_leaf=3, min_samples_split=3,
#                                         n_estimators=50, n_jobs=-1))])
# 최적의 매개변수 :  {'svm__max_depth': 12, 'svm__max_leaf_nodes': 100, 'svm__min_samples_leaf': 3, 'svm__min_samples_split': 3, 'svm__n_estimators': 50, 'svm__n_jobs': -1}
# 최종 정답률     :  0.05505129721529389


# RandomizedSearchCV
# [Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:    8.5s finished
# 최적의 모델    :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('svm',
#                  RandomForestClassifier(max_depth=10, max_leaf_nodes=90,
#                                         min_samples_leaf=7, n_estimators=200,
#                                         n_jobs=-1))])
# 최적의 매개변수 :  {'svm__n_jobs': -1, 'svm__n_estimators': 200, 'svm__min_samples_split': 2, 'svm__min_samples_leaf': 7, 'svm__max_leaf_nodes': 90, 'svm__max_depth': 10}
# 최종 정답률     :  0.28835365487416653


