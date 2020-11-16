import numpy as np
from keras.models import Sequential , Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터준비

# 2. 모델 구성
# input1 = Input(shape=(4,1))
# lstm1_1  = LSTM(10,activation='relu',name='lstm1_1')(input1)
# dense1_1 = Dense(10,activation='relu',name='dense1_1')(lstm1_1)
# dense1_2 = Dense(10,activation='relu',name='dense1_2')(dense1_1)
# dense1_3 = Dense(10,activation='relu',name='dense1_3')(dense1_2)
# output1 = Dense(1,activation='relu',name='output1')(dense1_2)

# model = Model(inputs=(input1),outputs=(output1))
model = Sequential()
model.add(LSTM(200,activation='relu',input_shape=(3,1),name='dense1_1'))
model.add(Dense(180,activation='relu',name='dense1_2'))
model.add(Dense(150,activation='relu',name='dense1_3'))
model.add(Dense(120,activation='relu',name='dense1_4'))
model.add(Dense(100,activation='relu',name='dense1_5'))
model.add(Dense(1,name='dense1_6'))

model.summary()

model.save('./save/keras28_1.h5')
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# dense1_1 (Dense)             (None, 200)               800
# _________________________________________________________________
# dense1_2 (Dense)             (None, 180)               36180
# _________________________________________________________________
# dense1_3 (Dense)             (None, 150)               27150
# _________________________________________________________________
# dense1_4 (Dense)             (None, 120)               18120
# _________________________________________________________________
# dense1_5 (Dense)             (None, 100)               12100
# _________________________________________________________________
# dense1_6 (Dense)             (None, 1)                 101
# =================================================================
# Total params: 94,451
# Trainable params: 94,451
# Non-trainable params: 0


#model.save('.//save//keras28_2.h5')
# model.save('.\save\keras28_3.h5')
# model.save('.\\save\\keras28_4.h5')


# 3. 





















