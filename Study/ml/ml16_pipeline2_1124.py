# 분류 
# 클래스파이어 모델들을 추출
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline

iris = pd.read_csv("./data/csv/iris_ys.csv",header=0,index_col=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=55,shuffle=True)

# parameters = [
#     {"svm__C":[1, 10, 100, 1000], "svm__kernel":["linear"]},
#     {"svm__C":[1, 10, 100, 1000], "svm__kernel":["rbf"]    , "svm__gamma":[0.001, 0.0001]},
#     {"svm__C":[1, 10, 100, 1000], "svm__kernel":["sigmoid"], "svm__gamma":[0.001, 0.0001]}
# ]
# # svc__ : pipeline을 엮었을 때 SVC()의 param이라고 명시

parameters = [
    {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["linear"]},
    {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["rbf"]    , "malddong__gamma":[0.001, 0.0001]},
    {"malddong__C":[1, 10, 100, 1000], "malddong__kernel":["sigmoid"], "malddong__gamma":[0.001, 0.0001]}
]



# 2. 모델
# pipe = make_pipeline(MinMaxScaler(), SVC())
pipe = Pipeline([("scaler",MinMaxScaler()), ("malddong",SVC())])


model = RandomizedSearchCV(pipe,parameters,cv=5)


# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가 예측
print("acc         : ",model.score(x_test,y_test))
print("best_prams_ : ",model.best_params_)