import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
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

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x, y, train_size=0.8, random_state=1,shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. model

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

acc_score = accuracy_score(y_real,y_predict.round())
print("acc_score :   ",acc_score)

# y_real :   [0 1 1 1 1 1 1 1 1 1]
# predict :  [1 1 1 1 1 1 1 1 1 1]
# model_score :  0.9459183673469388
# acc_score :    0.9