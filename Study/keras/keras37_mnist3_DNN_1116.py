# mnist를 DNN으로 바꿔라

# 1. 데이터준비
import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)

# x_train = x_train.reshape(60000,784)
# x_test = x_test.reshape(10000,784)
# print(x_test[0:10])
# print(y_test[0:10])

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (60000,10) (10000,10) -> 원핫인코딩이 적용되어 10이 추가됨
print("y_train[0] : ",y_train[0])                  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] -> 5를 원핫인코딩 적용시킨 모습

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test  = x_test.reshape(10000,784).astype('float32')/255. # minmax scaler의 효과
# x_test  = x_test.reshape(x_test.shape[0]) ->  가능하다.


# 2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(784,)))
model.add(Dense(80,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax')) 


# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train,epochs=1000,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측

loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss : \n",loss)
print("accuracy : \n",accuracy)

# x_pred = x_test[0:10]
# y_real = y_test[0:10]
# y_test_predict = model.predict(x_pred)
# print("예측값 : \n",y_test_predict)
# print("실제값 : \n",y_real)
x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])
# np.array(y_test_predict)
# print("예측값(복원전) : \n",y_test_predict)

# ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(-1,1)
ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
# print("y_test_predict : ",y_test_predict)
print("예측값 : ",ytp_recovery)

# y_real = np.array(np.argmax([y_test[0:10]]))
# y_real = np.argmax([y_test[0:10]])
# y_real = np.argmax([y_pred],axis=1).reshape(-1,1)
y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 : \n",y_real)

