
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
from keras.models import Model 
from keras.layers import Input, concatenate

input1 = Input(shape=(3,1))
LSTM1_1 = LSTM(100,activation='relu',name='LSTM1_1')(input1)
dense1_1 = Dense(90,activation='relu',name='dense1_1')(LSTM1_1)
dense1_2 = Dense(80,activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(70,activation='relu',name='dense1_3')(dense1_2)
dense1_4 = Dense(60,activation='relu',name='dense1_4')(dense1_3)
dense1_5 = Dense(50,activation='relu',name='dense1_5')(dense1_4)
dense1_6 = Dense(30,activation='relu',name='dense1_6')(dense1_5)
dense1_7 = Dense(10,activation='relu',name='dense1_7')(dense1_6)
output2 = Dense(1,name="output2")(dense1_7)

model = Model(inputs=(input1),outputs=output2)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')
model.fit(x,y,epochs=1000,batch_size=1)

# 4. 평가, 예측

x_input = x_input.reshape(1,3,1)
print(x_input.shape)
x_input_predict = model.predict(x_input)
print("x_input_predict : ",x_input_predict)


# x_input_predict :  [[80.092384]]
# loss: 0.0771




