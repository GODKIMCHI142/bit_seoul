# 다중분류

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape,x_test.shape) # (50000, 32, 32,3) (10000, 32, 32,3)
print(y_train.shape,y_test.shape) # (50000,1) (10000,1)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = x_train.reshape(50000,32*32*3)
scaler.fit(x_train)

# fit한 결과로 transform
x_train = scaler.transform(x_train)

x_test = x_test.reshape(10000,32*32*3)
x_test  = scaler.transform(x_test) 

x_train = x_train.reshape(50000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)
# x_train = x_train.astype('float32')/255.
# x_test  = x_test.astype('float32')/255. # minmax scaler의 효과



# one hot encoding
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (50000,10) (10000,10) -> 원핫인코딩이 적용되어 10이 추가됨
print(y_train[0])                  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] -> 5를 원핫인코딩 적용시킨 모습

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(60,(3,3),input_shape=(32,32,3),activation='relu')) 
model.add(Conv2D(50,(3,3),activation='relu'))     
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2     
model.add(Conv2D(40,(3,3),activation='relu'))                      
model.add(Conv2D(30,(3,3),activation='relu'))                      
model.add(Dropout(0.2))                
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2  
model.add(Conv2D(20,(3,3),activation='relu')) 
model.add(Dropout(0.2))
model.add(Flatten())     
model.add(Dense(10,activation='relu'))          
model.add(Dense(100,activation='softmax'))                       
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
model.fit(x_train,y_train, epochs=1000,batch_size=32,verbose=1,
          validation_split=0.2,callbacks=[es])
print("fitend")
# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss :",loss)
print("accuracy :",accuracy)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
print("예측값 :",ytp_recovery)

y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 :",y_real)








