import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
x, y = load_diabetes(return_X_y=True)
# x, y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x, y, train_size=0.8, random_state=1,shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = RandomForestClassifier()
# model = KNeighborsRegressor()
# model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)
x_pred = x_test[:10]
y_real = y_test[:10]
# 4. 평가 예측
y_predict = model.predict(x_pred)
print("y_real :  ",y_real)
print("predict : ",y_predict.round())

model_score = model.score(x_test,y_test)
print("model_score : ",model_score)

acc_score = accuracy_score(y_real,y_predict.round())
print("acc_score :   ",acc_score)
# 분류 => accuracy_score
# 회귀 => r2_score

# R2
from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2_score(y_real,y_predict))

# ==== 분류

# LinearSVC
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [219.  85. 113.  85. 150. 170. 263. 178. 101.  50.]
# model_score :  0.0
# acc_score :    0.0

# SVC
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [ 90.  72. 128.  72. 150. 275. 275.  53.  53.  72.]
# model_score :  0.011235955056179775
# acc_score :    0.0

# KNeighborsClassifier
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [ 64.  37.  55.  37. 122.  91. 275.  44.  47.  50.]
# model_score :  0.0
# acc_score :    0.0

# RandomForestClassifier
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [ 64.  40. 140.  77. 107. 110. 275. 118. 124. 113.]
# model_score :  0.0
# acc_score :    0.0

# ==== 회귀

# KNeighborsRegressor
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [132.6 105.  148.   64.8 188.  160.8 223.2 112.2 117.2 105.2]
# model_score :  0.2602895052339007
# acc_score :    0.0
# R2 :  0.46478639007471023

# RandomForestRegressor
# y_real :   [ 78. 152. 200.  59. 311. 178. 332. 132. 156. 135.]
# predict :  [126.03 102.72 160.46  75.96 156.16 254.02 232.27  89.18 155.19  94.19]
# model_score :  0.30390181303474784
# acc_score :    0.0
# R2 :  0.29832587546033273