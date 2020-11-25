# pca로 축소해서 모델을 완성하시오
# 1. 0.95이상
# 2. 1 이상
# mnist dnn과 loss / acc를 비교

# mnist를 DNN으로 바꿔라

# 1. 데이터준비
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) 
print(y_train.shape,y_test.shape) 



x = np.append(x_train, x_test, axis=0)
x = x.reshape(x.shape[0],(x.shape[1]*x.shape[2])) # 7만, 784
y = np.append(y_train, y_test, axis=0)

# cumsum
pca = PCA()
pca.fit(x)
# 중요도가 높은순서대로 바뀐다.
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cumsum : 누적된 합
d = np.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=d)

x = pca.fit_transform(x)
print(x.shape)# 70000, 154
print(y.shape)# 70000, 154




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

# 2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(50,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax')) 


# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=50, mode='auto')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train,epochs=1000,validation_split=0.2,callbacks=[es], batch_size=32)

# 4. 평가, 예측

loss,accuracy = model.evaluate(x_test,y_test,batch_size=32)
print("loss : ",loss)
print("accuracy : ",accuracy)

x_pred = x_test[0:10]
y_pred = y_test[0:10]


# Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)
y_test_predict = model.predict([x_pred])

ytp_recovery = np.argmax(y_test_predict,axis=1).reshape(10)
print("예측값 : ",ytp_recovery)


y_real = np.argmax(y_pred,axis=1).reshape(10)
print("실제값 : ",y_real)



# 0.95 이상
# loss :  0.2340102642774582
# accuracy :  0.9643571376800537
# 예측값 :  [6 2 7 5 7 6 3 5 1 3]
# 실제값 :  [6 2 7 5 7 6 3 5 1 3]