
import numpy as np
# 1. 데이터준비

x = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],[7,8,9],
              [8,9,10],[9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]]) # (13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13,)
x_input = np.array([50,60,70]) # (3, )

print(x.shape)
print(y.shape)
print(x_input.shape)

x = x.reshape(13,3,1)
print(x.shape)
# y = y.reshape(13)
# print(y.shape)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(15,activation='relu',input_shape=(3,1),return_sequences = True))
model.add(LSTM(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(1))

# ValueError: Input 0 of layer lstm_1 is incompatible with the layer:
# expected ndim=3, found ndim=2. Full shape received: [None, 500]

model.summary()
# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')

from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss',patience=10, mode='min')
early_stopping = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.fit(x,y,epochs=10000,batch_size=1,callbacks=[early_stopping])


# 4. 평가, 예측

x_input = x_input.reshape(1,3,1)
print(x_input.shape)
x_input_predict = model.predict(x_input)
print("x_input_predict : ",x_input_predict)


# x_input_predict :  [[80.092384]]
# loss: 0.0771




