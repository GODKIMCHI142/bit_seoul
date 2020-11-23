import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
dataset = load_boston()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']


from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x, y, train_size=0.8, random_state=1,shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# x_train = scaler.transform(x_train).astype("float")
# x_test = scaler.transform(x_test).astype("float")
# y_train = y_train.astype("float")
# y_test = y_test.astype("float")

# 2. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = RandomForestClassifier()
# model = KNeighborsRegressor()
# model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)
x_pred = x_test[:10]
y_real = y_test[:10]
# 4. 평가 예측
y_predict = model.predict(x_pred)
print("y_real :  ",y_real)
print("predict : ",y_predict)

model_score = model.score(x_test,y_test)
print("model_score : ",model_score)

# acc_score = accuracy_score(y_real,y_predict.astype("float"))
# print("acc_score :   ",acc_score)
# 분류 => accuracy_score
# 회귀 => r2_score

# R2
from sklearn.metrics import r2_score
# r2 = r2_score(y_test,x_test_predict)
print("R2 : ",r2_score(y_real,y_predict))

# ==== 분류

# LinearSVC
# ValueError: Unknown label type: 'continuous'

# SVC
# ValueError: Unknown label type: 'continuous'

# KNeighborsClassifier
# ValueError: Unknown label type: 'continuous'

# RandomForestClassifier
# ValueError: Unknown label type: 'continuous'


# ==> 이진분류, 다중분류만 지원한다.



# ==== 회귀

# KNeighborsRegressor
# y_real :   [28.2 23.9 16.6 22.  20.8 23.  27.9 14.5 21.5 22.6]
# predict :  [29.24 24.32 20.22 21.82 18.36 18.34 28.36 17.68 21.8  26.04]
# model_score :  0.8105457731109476
# R2 :  0.6107517249727634

# RandomForestRegressor
# y_real :   [28.2 23.9 16.6 22.  20.8 23.  27.9 14.5 21.5 22.6]
# predict :  [29.595 27.134 19.533 20.548 19.059 19.658 27.522 19.087 20.497 23.333]
# model_score :  0.9164651289492116
# R2 :  0.6365913932937897