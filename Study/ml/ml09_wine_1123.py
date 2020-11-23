import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
dataset = load_wine()
x = dataset.data
y = dataset.target
print(dataset.feature_names) # x의 컬럼명
# ['alcohol',        'malic_acid',      'ash',        'alcalinity_of_ash', 
# 'magnesium',       'total_phenols',   'flavanoids', 'nonflavanoid_phenols', 
# 'proanthocyanins', 'color_intensity', 'hue',        'od280/od315_of_diluted_wines', 'proline']

print(dataset.target_names) # y의 컬럼명
# ['class_0' 'class_1' 'class_2']


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
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2 1 0 1 0 2 1 0 2 1]
# model_score :  1.0
# acc_score :    1.0
# R2 :  1.0

# SVC
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2 1 0 1 0 2 1 0 2 1]
# model_score :  0.9722222222222222
# acc_score :    1.0
# R2 :  1.0

# KNeighborsClassifier
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2 1 0 1 0 2 1 0 2 1]
# model_score :  0.9722222222222222
# acc_score :    1.0
# R2 :  1.0

# RandomForestClassifier
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2 1 0 1 0 2 1 0 2 1]
# model_score :  0.9722222222222222
# acc_score :    1.0
# R2 :  1.0

# ==== 회귀

# KNeighborsRegressor
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2. 1. 0. 1. 0. 2. 1. 0. 2. 1.]
# model_score :  0.9426151930261519
# acc_score :    1.0
# R2 :  0.96

# RandomForestRegressor
# y_real :   [2 1 0 1 0 2 1 0 2 1]
# predict :  [2. 1. 0. 1. 0. 2. 1. 0. 2. 1.]
# model_score :  0.9435073474470734
# acc_score :    1.0
# R2 :  0.9839833333333333

