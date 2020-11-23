# winequality-white.csv 
# RandomForest로 모델 만들기
# 분류모델인듯

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score




# 1. data
dataset = pd.read_csv("./data/csv/winequality-white.csv",header=0,index_col=None,sep=";")

print(dataset)
print(dataset.shape)

datasets = pd.DataFrame(dataset)
datasets = datasets.values
print(dataset.shape)
print(type(datasets))
x = datasets[:,:11]
y = datasets[:,11:]
print(x.shape)
print(y.shape)
print(x)
print(y)


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
# model = RandomForestClassifier()
# model = KNeighborsRegressor()
model = RandomForestRegressor()

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

acc_score = accuracy_score(y_real,y_predict.round())
print("acc_score :   ",acc_score)
# 분류 => accuracy_score
# 회귀 => r2_score

# R2
# from sklearn.metrics import r2_score
# print("R2 : ",r2_score(y_real,y_predict))

# ==== 분류

# LinearSVC
# model_score :  0.5265306122448979
# acc_score :    0.2

# SVC
# model_score :  0.5561224489795918
# acc_score :    0.3

# KNeighborsClassifier
# model_score :  0.5571428571428572
# acc_score :    0.2

# RandomForestClassifier
# model_score :  0.6836734693877551
# acc_score :    0.4

# ==== 회귀

# KNeighborsRegressor
# model_score :  0.35631420985015383
# acc_score :    0.2

# RandomForestRegressor
# model_score :  0.5219016183504787
# acc_score :    0.4


