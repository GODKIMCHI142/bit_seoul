# 1. 데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # (4,3)
y = np.array([4,5,6,7])                         # (4, )

print("x.shape, y.shape : ",x.shape,y.shape)

x = x.reshape(x.shape[0] , x.shape[1] , 1)
# x = x.reshape(4,3,1)
print("x.shape : ",x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,shuffle=True, random_state=1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
# model.add(LSTM(100,activation='relu',input_shape=(3,1)))
model.add(LSTM(50,activation='relu',input_length=3,input_dim=1))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x,y,epochs=200,batch_size=1)


# 4. 평가, 예측

# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

x_input = np.array([5,6,7]) # (3, ) -> (1,3,1)
x_input = x_input.reshape(1,3,1)

x_input_predict = model.predict(x_input)
print("x_input_preidct : ",x_input_predict)

# y_test_predict = model.predict(x_test)
# print("y_test : ",y_test)
# print("y_test_predict : ",y_test_predict)















