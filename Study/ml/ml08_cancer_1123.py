import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# 1. data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']


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
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [1 0 1 0 0 0 0 0 1 1]
# model_score :  0.9736842105263158
# acc_score :    1.0
# R2 :  1.0

# SVC
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [1 0 1 0 0 0 0 0 1 1]
# model_score :  0.9736842105263158
# acc_score :    1.0
# R2 :  1.0

# KNeighborsClassifier
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [1 0 1 0 1 0 0 0 1 1]
# model_score :  0.956140350877193
# acc_score :    0.9
# R2 :  0.5833333333333333

# RandomForestClassifier
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [1 0 1 0 1 0 0 0 1 1]
# model_score :  0.956140350877193
# acc_score :    0.9
# R2 :  0.5833333333333333

# ==== 회귀

# KNeighborsRegressor
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [1. 0. 1. 0. 1. 0. 0. 0. 1. 1.]
# model_score :  0.819047619047619
# acc_score :    0.9
# R2 :  0.5499999999999999

# RandomForestRegressor
# y_real :   [1 0 1 0 0 0 0 0 1 1]
# predict :  [0. 0. 1. 0. 1. 0. 0. 0. 1. 1.]
# model_score :  0.8278690476190477
# acc_score :    0.8
# R2 :  0.7173749999999999