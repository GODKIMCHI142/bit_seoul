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
model.add(Dense(1,input_shape=(2,),activation="sigmoid"))

# hidden layer가 없어 아무리 훈련을 하여도 정확도에 기대값은 낮다.

# 3. 훈련
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_data,y_data,batch_size=1,epochs=300)

# 4. 평가, 예측
result = model.evaluate(x_data,y_data)
print("loss : ",result[0])
print("acc : ",result[1])

y_predict = model.predict(x_data)
print("predict : ",y_predict)

# acc_score = accuracy_score(y_data,y_predict)
# print("acc_score : ",acc_score)
# => 고치기


# model_score = model.score(x_data,y_data)
# print("model_score : ",model_score)
# loss :  0.7144027948379517
# acc :  0.5