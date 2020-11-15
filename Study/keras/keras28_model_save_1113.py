import numpy as np
from keras.models import Sequential , Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터준비

# 2. 모델 구성
input1 = Input(shape=(4,1))
lstm1_1  = LSTM(10,activation='relu',name='lstm1_1')(input1)
dense1_1 = Dense(10,activation='relu',name='dense1_1')(lstm1_1)
dense1_2 = Dense(10,activation='relu',name='dense1_2')(dense1_1)
dense1_3 = Dense(10,activation='relu',name='dense1_3')(dense1_2)
output1 = Dense(1,activation='relu',name='output1')(dense1_2)

model = Model(inputs=(input1),outputs=(output1))
model.summary()

model.save('./save/keras28_1.h5')
#model.save('.//save//keras28_2.h5')
# model.save('.\save\keras28_3.h5')
# model.save('.\\save\\keras28_4.h5')


# 3. 





















