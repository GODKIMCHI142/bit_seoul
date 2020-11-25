# OneHotEncoding

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from sklearn.decomposition import PCA



(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape,x_test.shape) # (50000, 32, 32,3) (10000, 32, 32,3)
print(y_train.shape,y_test.shape) # (50000,1) (10000,1)


x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0],(x.shape[1]*x.shape[2]*x.shape[3])) 
y = np.append(y_train, y_test, axis=0)

# cumsum
pca = PCA()
pca.fit(x) # 중요도가 높은순서대로 바뀐다.
cumsum = np.cumsum(pca.explained_variance_ratio_) 

d = np.argmax(cumsum >= 0.95) + 1
print(d) # 202
pca = PCA(n_components=d)
x = pca.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8, random_state=1)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

# fit한 결과로 transform
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# 2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Dense(200,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(150,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='softmax')) 

model.summary()


# 3. 컴파일, 훈련

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=20, mode='auto')

# Compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit
model.fit(x_train,y_train, epochs=1000,batch_size=64,verbose=1,
          validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
loss,accuracy = model.evaluate(x_test,y_test,batch_size=64)
print("loss     : ",loss)
print("accuracy : ",accuracy)

# predict
x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
print("예측값 : ",ytp_recovery)

y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 : ",y_real)





