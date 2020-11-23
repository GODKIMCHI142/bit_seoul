import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
# x, y = load_iris(return_X_y=True)
x, y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x, y, train_size=0.8, random_state=1,shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가 예측
y_predict = model.predict(x_test)
print("y_real :  ",y_test)
print("predict : ",y_predict.round())

model_score = model.score(x_test,y_test)
print("model_score : ",model_score)

acc_score = accuracy_score(y_test,y_predict.round())
print("acc_score :   ",acc_score)
# 분류 => accuracy_score
# 회귀 => r2_score

# ==== 분류

# LinearSVC
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0 1 1 0 2 2 2 0 0 2 1 0 2 1 1 0 1 2 0 0 1 2 2 0 2 1 0 0 1 2]
# model_score :  0.8666666666666667
# acc_score :    0.8666666666666667

# SVC
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2]
# model_score :  0.9666666666666667
# acc_score :    0.9666666666666667

# KNeighborsClassifier
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2]
# model_score :  0.9666666666666667
# acc_score :    0.9666666666666667

# RandomForestClassifier
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2]
# model_score :  0.9666666666666667
# acc_score :    0.9666666666666667

# ==== 회귀

# KNeighborsRegressor
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0.  1.  1.  0.  2.  1.4 2.  0.  0.  2.  1.  0.  2.  1.  1.  0.  1.  1.
#  0.  0.  1.  1.  1.6 0.  2.  1.  0.  0.  1.2 1.6]
# model_score :  0.9554639175257732
# acc_score :    0.9666666666666667

# RandomForestRegressor
# y_real :   [0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 1 0 2 1 0 0 1 2]
# predict :  [0.   1.01 1.   0.   2.   1.04 2.   0.   0.   2.   1.   0.   2.   1.02
#  1.02 0.   1.   1.02 0.   0.   1.   1.02 2.   0.   1.99 1.   0.   0.
#  1.   1.98]
# model_score :  0.9379092783505155
# acc_score :    0.9666666666666667