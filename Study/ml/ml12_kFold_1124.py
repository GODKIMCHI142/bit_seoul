import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


# 1. data

iris = pd.read_csv("./data/csv/iris_ys.csv",header=0,index_col=0)
x = iris.iloc[:,:4]
y = iris.iloc[:,4]

print(x.shape)
print(y.shape)


x_train, x_test , y_train  , y_test = train_test_split(x, y, train_size=0.8, random_state=24,shuffle=True)

# scaler  = StandardScaler()
# scaler.fit(x)
# x_train = scaler.transform(x_train)
# x_test  = scaler.transform(x_test)





# 2. 모델

kfold = KFold(n_splits=5,shuffle=True,random_state=24)
# 5조각을 내고 그것들을 섞어서 하겠다.

model = SVC()

s = cross_val_score(model,x_train,y_train,cv=kfold)

print("scores : ",s)
# scores :  [0.95833333 1.         0.95833333 0.875      1.        ]


# 3. 훈련
# model.fit(x_train,y_train)
# x_pred = x_test[:10]
# y_real = y_test[:10]

# # 4. 평가 예측
# y_predict = model.predict(x_pred)
# print("y_real :  ",y_real)
# print("predict : ",y_predict)

# model_score = model.score(x_test,y_test)
# print("model_score : ",model_score)

# acc_score = accuracy_score(y_real,y_predict.round())
# print("acc_score :   ",acc_score)









