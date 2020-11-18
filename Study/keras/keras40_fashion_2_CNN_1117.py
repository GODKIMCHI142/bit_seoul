# 다중분류
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




x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test  = x_test.reshape(10000,28,28,1).astype('float32')/255. # minmax scaler의 효과
print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 

# x_test  = x_test.reshape(x_test.shape[0]) ->  가능하다.

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Conv2D(10,(2,2),padding='same',input_shape=(28,28,1))) # 28,28,60
model.add(Conv2D(10,(2,2),padding='valid'))                      # 27,27,50
model.add(Conv2D(10,(3,3)))                                      # 25,25,40
model.add(Conv2D(10,(2,2),strides=2))                            # 12,12,30
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2   # 6 ,6 ,30
model.add(Flatten())                                             # 1080,
model.add(Dense(10,activation='relu'))                           # 20,
model.add(Dense(10,activation='softmax'))                        # 10,
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
model.fit(x_train,y_train, epochs=1000,batch_size=32,verbose=1,validation_split=0.3,callbacks=[es])

print("fit 끝")

# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=16)
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



# Epoch 96/1000
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.0491 - accuracy: 0.9829 - val_loss: 1.1699 - val_accuracy: 0.8733
# Epoch 97/1000
# 1500/1500 [==============================] - 5s 4ms/step - loss: 0.0476 - accuracy: 0.9825 - val_loss: 1.1298 - val_accuracy: 0.8838
# 313/313 [==============================] - 1s 2ms/step - loss: 1.2395 - accuracy: 0.8763
# loss : 1.2395468950271606
# accuracy : 0.8762999773025513
# 예측값 : [9 2 1 1 6 1 4 6 5 7]
# 실제값 : [9 2 1 1 6 1 4 6 5 7]


