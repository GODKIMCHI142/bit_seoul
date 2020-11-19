# LSTM 대신 Conv1d를 사용하자
# CNN으로

# 1. 데이터준비
import numpy as np


x_train = np.load("./data/mnist_x_train.npy")
x_test  = np.load("./data/mnist_x_test.npy")
y_train = np.load("./data/mnist_y_train.npy")
y_test  = np.load("./data/mnist_y_test.npy")

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (60000, 28, 28)
# (10000, 28, 28)
# (60000,)
# (10000,)

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000,10) (10000,10) -> 원핫인코딩이 적용되어 10이 추가됨
print("y_train[0] : ",y_train[0])                  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] -> 5를 원핫인코딩 적용시킨 모습

x_train = x_train.astype('float32')/255.
x_test  = x_test.astype('float32')/255. # minmax scaler의 효과
# x_test  = x_test.reshape(x_test.shape[0]) ->  가능하다.


# 2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(10,3,activation='relu',input_shape=(28,28)))
model.add(Conv1D(10,3,activation='relu'))
model.add(Conv1D(10,3,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10,activation='relu')) 
model.add(Dense(10,activation='relu')) 
model.add(Dense(10,activation='softmax')) 
model.summary()

# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=5, mode='auto')

# Compile
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

# fit
model.fit(x_train,y_train,epochs=30,batch_size=32,validation_split=0.2)

# 4. 평가, 예측

loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("accuracy : ",accuracy)

x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
ytp_recovery   = np.argmax(y_test_predict,axis=1).reshape(10)
y_real         = np.argmax(y_pred,axis=1).reshape(10)

print("예측값 : ",ytp_recovery)
print("실제값 : ",y_real)


# Epoch 30/30
# loss :  0.07240833342075348
# accuracy :  0.9779000282287598
# 예측값 :  [7 2 1 0 4 1 4 9 5 9]
# 실제값 :  [7 2 1 0 4 1 4 9 5 9]



