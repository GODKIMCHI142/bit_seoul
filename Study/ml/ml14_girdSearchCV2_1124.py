# 유방암 데이터 
# 모델 : RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66,shuffle=True)

# 2. 모델
kfold = KFold(n_splits=5,shuffle=True)
# model = SVC()
model = GridSearchCV(RandomForestClassifier(), parameters2, cv=kfold)


# 3. 훈련
model.fit(x_train, y_train)
ss = model.best_estimator_
# 4. 평가 예측
print("최적의 매개변수 : ",model.best_estimator_)

y_predict = model.predict(x_test)
print("최종 정답률     : ",accuracy_score(y_test,y_predict))



# 최적의 매개변수 :  RandomForestClassifier()
# 최종 정답률     :  0.9707602339181286


# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=3,
#                        n_jobs=-1)
# 최종 정답률     :  0.9590643274853801
