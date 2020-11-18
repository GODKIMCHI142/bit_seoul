# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape) 
print(y.shape) 

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

# fit한 결과로 transform
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8)

print(x.shape)
print(y.shape)
print(x_train.shape) # 455,30
print(x_test.shape)  # 114,30
print(y_train.shape) # 455,
print(y_test.shape)  # 114,

x_train = x_train.reshape(455,6,5,1)
x_test  = x_test.reshape(114,6,5,1)
y_train = y_train.reshape(455,1)
y_test  = y_test.reshape(114,1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(100,(2,2),padding="same",input_shape=(6,5,1),activation='relu')) 
model.add(Flatten())
model.add(Dense(90,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(51,activation='softmax'))                       
model.summary()

# 3. 컴파일 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=100, mode='auto')

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=1000,batch_size=10,validation_split=0.3,callbacks=[es],verbose=1)

print("fit end")
# 4. 평가 예측
loss, acc = model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",loss)
print("acc : ",acc)
# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]

y_test_predict = model.predict(x_pred)

print("예측값 :",y_test_predict)
print("실제값 :",y_pred)



