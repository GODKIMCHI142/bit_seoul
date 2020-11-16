import numpy as np

# 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Model
from keras.applications import VGG16 
from keras.layers import Dropout
from keras.layers import Input, concatenate


model = load_model("./save/keras28_1.h5")
model.summary()

input1 = Input(shape=(4,1))
new_dense  = model(input1)
dense1_1 = Dense(10,activation='relu',name='dense2_1')(new_dense)
dense1_2 = Dense(10,activation='relu',name='dense2_2')(dense1_1)
dense1_3 = Dense(10,activation='relu',name='dense2_3')(dense1_2)
output2_1 = Dense(1,activation='relu',name='output2_1')(dense1_3)

model2 = Model(inputs=input1,outputs=output2_1)
model2.summary()



