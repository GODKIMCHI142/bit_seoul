# OneHotEncoding

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)
# print(x_train[0])
print(y_test[0:10])

# plt.imshow(x_train[0])
# plt.show()

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000,10) (10000,10) -> 원핫인코딩이 적용되어 10이 추가됨
print(y_train[0])                  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] -> 5를 원핫인코딩 적용시킨 모습

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test  = x_test.reshape(10000, 28, 28, 1).astype('float32')/255. # minmax scaler의 효과
# x_test  = x_test.reshape(x_test.shape[0]) ->  가능하다.


# print(x_train[0]) # 최대값은 255이다. 명암을 0 ~ 255 값으로 나타내었다.

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(60,(2,2),padding='same',input_shape=(28,28,1))) # 28,28,60
model.add(Dropout(0.2))
model.add(Conv2D(50,(2,2),padding='valid'))                      # 27,27,50
model.add(Dropout(0.2))
model.add(Conv2D(40,(3,3)))                                      # 25,25,40
model.add(Dropout(0.2))
model.add(Conv2D(30,(2,2),strides=2))                            # 12,12,30
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2   # 6 ,6 ,30
model.add(Dropout(0.2))
model.add(Flatten())                                             # 1080,
model.add(Dense(30,activation='relu'))                           # 20,
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))                           # 20,
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))                        # 10,

model.summary()

# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=30, mode='auto')

# tensorboard
from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='graph',histogram_freq=0, write_graph=True, write_images=True)

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit
model.fit(x_train,y_train, epochs=30,batch_size=32,verbose=1,
          validation_split=0.2,callbacks=[es,tb])

# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss : \n",loss)
print("accuracy : \n",accuracy)
# print("x_test[0] : \n",x_test[0])

x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
# np.array(y_test_predict)

# ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(-1,1)
ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
# print("y_test_predict : ",y_test_predict)
print("ytp_recovery : ",ytp_recovery)

# y_real = np.array(np.argmax([y_test[0:10]]))
# y_real = np.argmax([y_test[0:10]])
# y_real = np.argmax([y_pred],axis=1).reshape(-1,1)
y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 : \n",y_real)

