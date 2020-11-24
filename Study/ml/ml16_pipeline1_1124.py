# 분류 
# 클래스파이어 모델들을 추출
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline


iris = pd.read_csv("./data/csv/iris_ys.csv",header=0,index_col=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=55,shuffle=True)

# scaler - pipeline

pipe = make_pipeline(MinMaxScaler(), SVC())
# 스케일러와 모델을 준비해놓은 상태
pipe.fit(x_train,y_train)
# 스케일러로 train을 fit하고 svc로 돌린다.
# pipeline이 과적합을 줄여준다.

print("acc : ",pipe.score(x_test,y_test))
# acc :  0.9666666666666667



# parameters = [
#     {"C":[1, 10, 100, 1000], "kernel":["linear"]},
#     {"C":[1, 10, 100, 1000], "kernel":["rbf"]    , "gamma":[0.001, 0.0001]},
#     {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
# ]


# # 2. 모델
# kfold = KFold(n_splits=5,shuffle=True)
# # model = SVC()
# model = GridSearchCV(SVC(), parameters, cv=kfold)


# # 3. 훈련
# model.fit(x_train, y_train)


# # 4. 평가 예측
# print("최적의 매개변수 : ",model.best_estimator_)

# y_predict = model.predict(x_test)
# print("최종 정답률     : ",accuracy_score(y_test,y_predict))