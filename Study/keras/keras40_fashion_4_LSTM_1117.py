# OneHotEncoding

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist



(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)




x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test  = x_test.reshape(10000,28*28).astype('float32')/255. # minmax scaler의 효과
print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 

# x_test  = x_test.reshape(x_test.shape[0]) ->  가능하다.

# 2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(LSTM(50,activation='relu',input_shape=(7*7,16)))
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax')) 
model.summary()


# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10, mode='auto')

# tensorboard
# from tensorflow.keras.callbacks import TensorBoard
# tb = TensorBoard(log_dir='graph',histogram_freq=0, write_graph=True, write_images=True)

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit
model.fit(x_train,y_train, epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss :",loss)
print("accuracy :",accuracy)

x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
print("예측값 :",ytp_recovery)

y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 :",y_real)






