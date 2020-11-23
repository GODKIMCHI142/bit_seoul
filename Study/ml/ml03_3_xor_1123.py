# 다중분류
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. data
x_data = [[0,0],[1,1],[0,1],[1,0]]
y_data = [0,0,1,1]

# 2. model
model = LinearSVC()

# 3. 훈련
model.fit(x_data,y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)
print("predict : ",y_predict)

acc_score = accuracy_score(y_data,y_predict)
print("acc_score : ",acc_score)

model_score = model.score(x_data,y_predict)
print("model_score : ",model_score)

