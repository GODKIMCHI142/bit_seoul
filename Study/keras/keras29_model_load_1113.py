import numpy as np

# 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Model
from keras.applications import VGG16 
from keras.layers import Dropout



model = load_model("./save/keras28_1.h5")
# dense2_1 = Dense(10,activation='relu',name='dense2_1')
# dense2_2 = Dense(10,activation='relu',name='dense2_2')(dense2_1)




# model.add(Dense(5,name='dense2_1',kernel_initializer='uniform'))
# model.add(Dense(4,name='dense2_2',kernel_initializer='uniform'))
# model.add(Dense(3,name='dense2_3'))
# model.add(Dense(2,name='dense2_4'))
# model.add(Dense(1,name='dense2_5'))
# model.add(Dense())

model.summary()
m_weight = model.get_weights()
print("m_weight : >>>> \n",m_weight)