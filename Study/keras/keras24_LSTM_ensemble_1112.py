### 실습 : 앙상블 모델을 만드시오. 목표는 85 ! 


import numpy as np
# 1. 데이터준비

x1 = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],[7,8,9],
              [8,9,10],[9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])            # (13,3)

x2 = np.array([[10,20,30],[20,30,40],[30,40,50],
              [40,50,60],[50,60,70],[60,70,80],[70,80,90],
              [80,90,100],[90,100,110],[100,110,120],
              [2,3,4],[3,4,5],[4,5,6]])                     # (13,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])            # (13,)
x1_predict = np.array([55,65,75])                           # (3, )
x2_predict = np.array([65,75,85])                           # (3, )

# new_x = np.array([x1_predict,x2_predict])
# print(new_x)
# print(type(new_x))
# print(new_x.shape)
# print(type(x1))
# print("x1 : >>> ",x1)
# print("x2 : >>> ",x2)
# print("new_x : >>> ",new_x)
# print(x1.shape)
# print(x2.shape)
# print(y.shape)
# print(x1_predict.shape)
# print(x2_predict.shape)
# # x1_predict = x1_predict.reshape(1,3)
# # x2_predict = x2_predict.reshape(1,3)
# print(x1_predict.shape,x2_predict.shape)
# print(x1_predict,x2_predict)
# xxx = np.array([x1_predict,x2_predict])
# print(xxx)

# x = x.reshape(13,3,1)
# print(x.shape)
# y = y.reshape(13)
# print(y.shape)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Model 
from keras.layers import Input, concatenate

# Model 1
input1 = Input(shape=(3,1))
LSTM1_1 = LSTM(1000,activation='relu',name='LSTM1_1')(input1)
dense1_1 = Dense(800,activation='relu',name='dense1_1')(LSTM1_1)
dense1_2 = Dense(600,activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(400,activation='relu',name='dense1_3')(dense1_2)
dense1_4 = Dense(200,activation='relu',name='dense1_4')(dense1_3)
output1 = Dense(100,name="output1")(dense1_4)

# Model 1
input2 = Input(shape=(3,1))
LSTM2_1 = LSTM(1,activation='relu',name='LSTM2_1')(input2)
# dense2_1 = Dense(15,activation='relu',name='dense2_1')(LSTM2_1)
# dense2_2 = Dense(15,activation='relu',name='dense2_2')(dense2_1)
# dense2_3 = Dense(15,activation='relu',name='dense2_3')(dense2_2)
# dense2_4 = Dense(15,activation='relu',name='dense2_4')(dense2_3)
output2 = Dense(1,name="output2")(LSTM2_1)

merge = concatenate([output1,output2])

# dense3_1 = Dense(1,activation='relu',name='dense3_1')(merge)
# dense3_2 = Dense(1,activation='relu',name='dense3_2')(dense3_1)
# dense3_3 = Dense(1,activation='relu',name='dense3_3')(dense3_2)
output1_1 = Dense(1,name="output1_1")(merge)


model = Model(inputs=[input1,input2],outputs=output1_1)

model.summary()



# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam',metrics='mae')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=100, mode='auto')
model.fit([x1,x2],y,epochs=1000,batch_size=13,callbacks=early_stopping)


# 4. 평가, 예측

x1_predict = x1_predict.reshape(1,3) 
x2_predict = x2_predict.reshape(1,3)
print("x1_predict : >>>> \n",x1_predict)
print("x2_predict : >>>> \n",x2_predict)
print(x1_predict.shape,x2_predict.shape)

# predict_data = model.predict(new_x)
# predict_data = model.predict([x1_predict,x2_predict])
predict_data = model.predict([x1_predict,x2_predict])
predict_data = predict_data.reshape(1,)
print("predict_data : >>>> \n",type(predict_data))
print("predict_data : >>>> \n",predict_data)
print("predict_data : >>>> \n",predict_data.shape)

# x_input_predict :  [[80.092384]]
# loss: 0.0771


# [[1,2,3],[4,5,6]]



# x1_predict : >>>>
#  [[55 65 75]]
# x2_predict : >>>>
#  [[65 75 85]]
# (1, 3) (1, 3)
# predict_data : >>>>
#  <class 'numpy.ndarray'>
# predict_data : >>>>
#  [85.05777]
# predict_data : >>>>
#  (1,)
# PS D:\workspace\Study>







