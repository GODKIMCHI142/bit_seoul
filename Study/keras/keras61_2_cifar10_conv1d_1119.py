# LSTM 대신 Conv1d를 사용하자
import numpy as np

x_train = np.load("./data/cifar10_x_train.npy")
x_test  = np.load("./data/cifar10_x_test.npy")
y_train = np.load("./data/cifar10_y_train.npy")
y_test  = np.load("./data/cifar10_y_test.npy")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# (50000, 32, 32, 3)
# (10000, 32, 32, 3)
# (50000, 1)
# (10000, 1)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = x_train.reshape(50000,32*32*3)
scaler.fit(x_train)

# fit한 결과로 transform
x_train = scaler.transform(x_train)

x_test = x_test.reshape(10000,32*32*3)
x_test  = scaler.transform(x_test) 

x_train = x_train.reshape(50000,64,16*3)
x_test = x_test.reshape(10000,64,16*3)

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(10,10,activation='relu',input_shape=(64,16*3)))
model.add(Conv1D(10,8,activation='relu'))
model.add(Conv1D(10,6,activation='relu'))
model.add(Conv1D(10,4,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation='relu')) 
model.add(Dense(10,activation='relu')) 
model.add(Dense(10,activation='softmax')) 
model.summary()


# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=10, mode='auto')

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit
model.fit(x_train,y_train, epochs=30,batch_size=32,verbose=1,
          validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("accuracy : ",accuracy)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
ytp_recovery   = np.argmax(y_test_predict,axis=1)
y_real         = np.argmax(y_pred,axis=1)

print("예측값 : ",ytp_recovery)
print("실제값 : ",y_real)


# loss : 1.4918396472930908
# accuracy : 0.45500001311302185
# 예측값 :  [3 8 8 1 4 6 1 6 4 1]
# 실제값 :  [3 8 8 0 6 6 1 6 3 1]