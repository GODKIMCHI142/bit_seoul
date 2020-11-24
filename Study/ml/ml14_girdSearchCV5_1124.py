# wine data
# 모델 : RandomForestClassifier

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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

wine = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")

x = wine.drop("quality", axis=1)
x = np.array(x)
y = wine["quality"]


print(x.shape) # (4898, 11)
print(y.shape) # (4898,)

newlist = []
for i in list(y):
    if   i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else: 
        newlist += [2]

y = newlist
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66,shuffle=True)

# 2. 모델
kfold = KFold(n_splits=5,shuffle=True)
# model = SVC()
model = GridSearchCV(RandomForestClassifier(), parameters2, cv=kfold, verbose=2)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측

print(model.cv_results_)
print("최적의 매개변수 : ",model.best_estimator_)
print(model.best_params_)


y_predict = model.predict(x_test)
# print("최종 정답률     : ",accuracy_score(y_test,y_predict))

from sklearn.metrics import r2_score
print("최종 정답률     : ",r2_score(y_test,y_predict))





