# 다중분류
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. data
x_data = [[0,0],[1,1],[0,1],[1,0]]
y_data = [0,0,1,1]

# 2. model
model = Sequential()
model.add(Dense(10,input_shape=(2,),activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# hidden layer에서 연산이 이루어지기 때문에 결과의 정확도를 높일수 있다.

# 3. 훈련
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_data,y_data,batch_size=1,epochs=50)

# 4. 평가, 예측
result = model.evaluate(x_data,y_data)
print("loss : ",result[0])
print("acc : ",result[1])

y_predict = model.predict(x_data)
print("predict : ",y_predict)

acc_score = accuracy_score(y_data,y_predict.round())
print("acc_score : ",acc_score)
# => 고치기

x = 3.14
x2 = 3.5
print(round(x))  # 3
print(round(x2)) # 4
# round() : 반올림


# model_score = model.score(x_data,y_data)
# print("model_score : ",model_score)

# loss :  0.6511015892028809
# acc :  1.0